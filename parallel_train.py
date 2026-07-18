#!/usr/bin/env python3
"""Parallel launcher for sliding walk-forward PPO training.

Spawns N independent Python processes, each training one fold on CPU.
Each fold writes to models/sliding/fold_N/ with zero cross-process
dependencies.  After all folds complete, stitches equity curves and runs the
consistency gate.

Usage:
    python parallel_train.py                       # 4 concurrent folds, all CPU
    python parallel_train.py --concurrency 6       # 6 concurrent folds
    python parallel_train.py --start-fold 8        # resume from fold 8
    python parallel_train.py --folds 1-5           # train only folds 1-5

Key design (verified from SB3 Issue #1931, PyTorch docs 2026):
- Each fold runs as a SEPARATE python process (not multiprocessing.Pool)
  to avoid PyTorch autograd engine contention.
- All folds use device="cpu" — PPO+MlpPolicy (256-neuron MLP, 22KB model) is
  1.2x faster on CPU than GPU (below the ~1500x1500 crossover where GPU wins).
  SB3 docs: "PPO is primarily intended to run on the CPU when not using CNN."
- CUDA_VISIBLE_DEVICES="" prevents any GPU context allocation (saves ~100MB RAM
  per process and avoids PyTorch autograd fork issues).
- OMP_NUM_THREADS + MKL_NUM_THREADS + torch.set_num_threads limit per-process
  thread pools to prevent CPU oversubscription.
- Each fold's train_single_fold() creates SubprocVecEnv with n_envs=4 env
  workers for parallel data collection during PPO rollout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# ── Config ────────────────────────────────────────────────────────────────────
_VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python3"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_ppo.py"
OUT_DIR = str(SCRIPT_DIR / "models")
TOTAL_FOLDS = 35  # sliding walk-forward produces ~35 folds
FOLD_TIME_ESTIMATE_SEC = 780  # ~13 min per fold (3M steps @ ~3800 fps)
DEFAULT_TOTAL_STEPS = 3_000_000
DEFAULT_SEED = 42
DEFAULT_N_ENVS = 4       # SubprocVecEnv workers per fold (CPU data collection)
DEFAULT_CONCURRENCY = 4  # max concurrent fold processes (RAM: ~6GB for 4×1.5GB)


def _get_fold_range(start_fold: int, end_fold: int, folds_str: Optional[str]) -> List[int]:
    """Parse fold range. Returns 1-based fold numbers."""
    if folds_str:
        # Parse "1-5,8,10-12" style
        folds = set()
        for part in folds_str.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                folds.update(range(int(lo), int(hi) + 1))
            else:
                folds.add(int(part))
        return sorted(f for f in folds if start_fold <= f <= end_fold)
    return list(range(start_fold, end_fold + 1))


def _check_fold_complete(fold_num: int) -> bool:
    """Check if a fold has already completed (has run_info.json + final model zip)."""
    fold_dir = Path(OUT_DIR) / "sliding" / f"fold_{fold_num}"
    has_run_info = (fold_dir / "run_info.json").exists()
    has_model = any(fold_dir.glob("ppo_*.zip")) or (fold_dir / "best_model" / "best_model.zip").exists()
    return has_run_info and has_model


def _launch_fold(
    fold_num: int,
    n_envs: int,
    concurrency: int,
    total_timesteps: int,
    seed: int,
    log_dir: Path,
) -> subprocess.Popen:
    """Launch a single fold as a separate OS process.

    Environment variables set in the subprocess:
    - CUDA_VISIBLE_DEVICES="" — prevent any GPU context allocation (saves RAM,
      avoids PyTorch autograd fork-safety issues).
    - OMP_NUM_THREADS / MKL_NUM_THREADS — limit thread pool per process so
      N concurrent folds don't oversubscribe CPU cores.
    """
    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--fold", str(fold_num),
        "--n-envs", str(n_envs),
        "--device", "cpu",
        "--num-workers", str(concurrency),
    ]

    log_file = log_dir / f"fold_{fold_num:02d}.log"
    log_fh = open(log_file, "w")

    # Per-process thread limit: 28 cores / N concurrent folds = ~7 threads each.
    # Each fold has 1 PPO process + 4 SubprocVecEnv workers = 5 processes,
    # but only the main process does gradient compute, so OMP threads mainly
    # affect NumPy/MKL in data loading and env stepping.
    n_cpus = os.cpu_count() or 28
    threads = max(1, n_cpus // concurrency)
    print(f"  [LAUNCH] fold {fold_num:>2}  device=cpu  n_envs={n_envs}  "
          f"threads={threads}  concurrency={concurrency}  → {log_file.name}")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",           # Force CPU-only (critical!)
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
        env=env,
    )
    return proc


def _monitor_batch(
    procs: dict[int, subprocess.Popen],
    log_dir: Path,
) -> dict[int, bool]:
    """Wait for all processes in batch to complete. Returns {fold: success}."""
    results = {}
    live = dict(procs)  # fold_num → Popen

    while live:
        for fold_num in list(live.keys()):
            proc = live[fold_num]
            retcode = proc.poll()
            if retcode is not None:
                del live[fold_num]
                success = retcode == 0
                status = "OK" if success else f"FAIL (exit={retcode})"
                print(f"  [DONE]  fold {fold_num:>2}  {status}")
                results[fold_num] = success

                # Print last few lines of log on failure
                if not success:
                    log_file = log_dir / f"fold_{fold_num:02d}.log"
                    if log_file.exists():
                        lines = log_file.read_text().strip().split("\n")
                        for line in lines[-5:]:
                            print(f"         {line}")

        if live:
            time.sleep(5)

    return results


def _compute_fold_metrics(equity_df, trades_df):
    """Compute fold metrics from equity + trades DataFrames (fallback for
    folds trained without fold_summary.json)."""
    from config import CFG
    from evaluate import full_report
    eq = equity_df if equity_df is not None else pd.DataFrame()
    tr = trades_df if trades_df is not None else pd.DataFrame()
    rep = full_report(eq, tr, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    return rep


def _stitch_results(folds: List[int]):
    """Stitch per-fold test equity curves into one continuous OOS track record
    and run the consistency gate.  Called after all folds complete."""
    import pandas as pd

    print(f"\n{'=' * 72}")
    print("  STITCHING RESULTS & RUNNING CONSISTENCY GATE")
    print(f"{'=' * 72}")

    summary_rows = []
    test_equities = []
    test_trade_logs = []

    for k in folds:
        fold_dir = Path(OUT_DIR) / "sliding" / f"fold_{k}"
        summary_file = fold_dir / "fold_summary.json"
        equity_file = fold_dir / "test_equity.csv"
        trades_file = fold_dir / "test_trades.csv"
        run_info_file = fold_dir / "run_info.json"

        # Load equity and trades (needed for stitching regardless of summary)
        eq_df = pd.read_csv(equity_file) if equity_file.exists() else None
        tr_df = pd.read_csv(trades_file) if trades_file.exists() else None

        if eq_df is not None:
            test_equities.append(eq_df)
        if tr_df is not None:
            test_trade_logs.append(tr_df)

        # Get or compute summary row
        if summary_file.exists():
            row = json.loads(summary_file.read_text())
        elif run_info_file.exists() and eq_df is not None:
            # Folds trained without fold_summary.json — compute metrics from equity
            rep = _compute_fold_metrics(eq_df, tr_df)
            row = {
                "fold": k,
                "test_return_pct": rep.get("total_return_pct"),
                "test_sharpe": rep.get("sharpe_like"),
                "test_profit_factor": rep.get("profit_factor"),
                "test_win_rate_pct": rep.get("win_rate_pct"),
                "test_max_dd_pct": rep.get("max_drawdown_pct"),
                "test_avg_r": rep.get("avg_r"),
                "test_n_trades": rep.get("n_trades"),
            }
            # Write summary for future use
            (fold_dir / "fold_summary.json").write_text(
                json.dumps(row, indent=2, default=str))
        else:
            print(f"  Fold {k}: no equity data (incomplete), skipping")
            continue

        summary_rows.append(row)
        ret = row.get("test_return_pct", 0) or 0
        pf = row.get("test_profit_factor", 0) or 0
        sharpe = row.get("test_sharpe", 0) or 0
        trades = row.get("test_n_trades", 0) or 0
        marker = "OK" if ret > 0 and pf > 1.0 else "  "
        print(f"  [{marker}] fold {k:>2}: return={ret:+6.1f}%  PF={pf:.2f}  "
              f"Sharpe={sharpe:+.2f}  trades={trades}")

    if not summary_rows:
        print("  No fold summaries found — nothing to stitch.")
        return

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(OUT_DIR) / "sliding_walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Stitch equity curves (compound)
    from config import CFG
    running = CFG.initial_equity
    parts = []
    for eq in test_equities:
        if eq is None or eq.empty or "equity" not in eq:
            continue
        s = eq["equity"].astype(float)
        scaled = s / CFG.initial_equity * running
        parts.append(scaled)
        running = float(scaled.iloc[-1])

    stitched = pd.concat(parts) if parts else pd.Series(dtype=float)
    stitched_df = stitched.to_frame("equity")
    stitched_path = Path(OUT_DIR) / "sliding_oos_equity.csv"
    stitched_df.to_csv(stitched_path)

    # Compute aggregate OOS metrics
    if test_trade_logs:
        all_trades = pd.concat(test_trade_logs, ignore_index=True)
    else:
        all_trades = pd.DataFrame()

    from evaluate import full_report
    oos = full_report(stitched_df, all_trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()

    print(f"\n  STITCHED out-of-sample track record ({len(stitched)} bars):")
    print(f"    total return : {oos.get('total_return_pct'):+.1f}%")
    print(f"    Sharpe-like  : {oos.get('sharpe_like'):+.2f}")
    print(f"    max drawdown : {oos.get('max_drawdown_pct'):+.1f}%")
    print(f"    profit factor: {oos.get('profit_factor'):.2f}   trades: {oos.get('n_trades')}")
    print(f"    equity → {stitched_path}")

    # Consistency gate
    sys.path.insert(0, str(SCRIPT_DIR))
    from train_ppo import _passes_consistency_gate, _finalize_deployment

    passed, detail = _passes_consistency_gate(
        summary, ret_col="test_return_pct",
        pf_col="test_profit_factor", sharpe_col="test_sharpe")
    print("\n  Consistency gate (on test windows):")
    for line in detail:
        print(f"    {line}")

    n_folds = len(folds)
    _finalize_deployment(OUT_DIR, n_folds, "sliding", passed)

    print(f"\n  Per-fold summary → {summary_path}")
    print(f"{'=' * 72}")


def main():
    ap = argparse.ArgumentParser(
        description="Parallel sliding walk-forward PPO training (CPU-only)")
    ap.add_argument("--start-fold", type=int, default=1,
                    help="First fold to train (1-based, default: 1)")
    ap.add_argument("--end-fold", type=int, default=None,
                    help="Last fold to train (default: all remaining)")
    ap.add_argument("--folds", type=str, default=None,
                    help="Fold range: '1-5' or '1,3,5' or '8-12,15'")
    ap.add_argument("--concurrency", type=int, default=None,
                    help=f"Max concurrent fold processes (default: {DEFAULT_CONCURRENCY}). "
                         f"With 30GB RAM, 4-5 is safe (each fold uses ~1.5GB).")
    ap.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
                    help=f"Total timesteps per fold (default: {DEFAULT_TOTAL_STEPS:,})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip folds that already have results (default: True)")
    args = ap.parse_args()

    print("=" * 72)
    print("  PARALLEL SLIDING WALK-FORWARD TRAINING (CPU-only)")
    print("=" * 72)

    # Hardware info
    n_cpus = os.cpu_count() or 28
    print(f"  Hardware: {n_cpus} CPU cores")

    # Concurrency (CPU-only — no GPU workers)
    concurrency = args.concurrency or DEFAULT_CONCURRENCY
    # Sanity: with SubprocVecEnv(n_envs=4), each fold spawns ~5 processes
    # (1 PPO + 4 env workers).  concurrency × 5 should be ≤ 3× n_cpus
    # to avoid severe oversubscription.
    max_safe = max(2, n_cpus // 4)  # conservative: ~7 for 28 cores
    if concurrency > max_safe:
        print(f"  ⚠ Concurrency {concurrency} > recommended max {max_safe} for {n_cpus} cores.")
        print(f"    Proceeding anyway — OMP/MKL threads will be throttled.")
    threads_per_fold = max(1, n_cpus // concurrency)

    print(f"  Concurrency: {concurrency} folds × {DEFAULT_N_ENVS} envs/fold")
    print(f"  Threads per fold process: {threads_per_fold} (OMP_NUM_THREADS)")
    print(f"  Total steps/fold: {args.total_steps:,}")
    print(f"  Device: CPU (all folds — PPO+MlpPolicy is 1.2x faster on CPU)")

    # Determine fold list
    n_folds = TOTAL_FOLDS
    end_fold = args.end_fold or n_folds
    folds = _get_fold_range(args.start_fold, end_fold, args.folds)
    print(f"  Folds to train: {folds[0]}-{folds[-1]} ({len(folds)} folds)")

    # Skip existing
    if args.skip_existing:
        before = len(folds)
        folds = [f for f in folds if not _check_fold_complete(f)]
        skipped = before - len(folds)
        if skipped:
            print(f"  Skipped {skipped} already-complete folds")

    if not folds:
        print("\n  All folds already complete! Nothing to do.")
        return

    # Estimate time
    est_per_fold = FOLD_TIME_ESTIMATE_SEC
    n_batches = (len(folds) + concurrency - 1) // concurrency
    est_total = n_batches * est_per_fold
    print(f"  Estimated time: {len(folds)} folds / {concurrency} concurrent = "
          f"{n_batches} batches × ~{est_per_fold / 60:.0f} min = "
          f"~{est_total / 60:.0f} min total")
    print(f"{'=' * 72}\n")

    # Create log directory
    log_dir = SCRIPT_DIR / "logs" / "parallel"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Main loop: process folds in batches ────────────────────────────────
    t_start = time.time()
    all_results = {}
    batch_num = 0

    for i in range(0, len(folds), concurrency):
        batch = folds[i:i + concurrency]
        batch_num += 1
        n_batches = (len(folds) + concurrency - 1) // concurrency
        print(f"── Batch {batch_num}/{n_batches}: folds {batch} ──")

        # Launch all workers in this batch (all CPU)
        procs = {}
        for fold_num in batch:
            proc = _launch_fold(
                fold_num=fold_num,
                n_envs=DEFAULT_N_ENVS,
                concurrency=concurrency,
                total_timesteps=args.total_steps,
                seed=args.seed,
                log_dir=log_dir,
            )
            procs[fold_num] = proc

        # Monitor
        batch_results = _monitor_batch(procs, log_dir)
        all_results.update(batch_results)

        # Check for failures — don't abort, just log
        failures = [f for f, ok in batch_results.items() if not ok]
        if failures:
            print(f"  ⚠ Batch {batch_num} had failures: {failures}")
            print(f"    (Will continue with remaining batches. Re-run to retry failures.)")

        elapsed = time.time() - t_start
        done = len(all_results)
        remaining = len(folds) - done
        if remaining > 0:
            eta = elapsed / done * remaining if done else 0
            print(f"  Progress: {done}/{len(folds)} done, "
                  f"elapsed {elapsed / 60:.1f} min, ETA ~{eta / 60:.1f} min\n")

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    successes = sum(1 for ok in all_results.values() if ok)
    failures = sum(1 for ok in all_results.values() if not ok)

    print(f"\n{'=' * 72}")
    print(f"  TRAINING COMPLETE — {successes}/{len(all_results)} folds succeeded"
          f"  ({failures} failures)")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"{'=' * 72}")

    # Stitch results if we have enough successful folds
    completed_folds = sorted(f for f, ok in all_results.items() if ok)
    if completed_folds:
        _stitch_results(completed_folds)

    if failures:
        print(f"\n  ⚠ {failures} fold(s) failed. Check logs/parallel/fold_XX.log for details.")
        print(f"    Re-run with --folds to retry specific folds.")
        sys.exit(1)


if __name__ == "__main__":
    main()
