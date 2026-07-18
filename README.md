# XAUUSD Reinforcement Learning Trading Pipeline

A reproducible walk-forward research pipeline for training PPO agents on XAUUSD (Gold) minute-bar data using bracket-order actions (direction + stop-loss + take-profit).

**Status**: Research prototype — not for live trading.

> **Based on**: [ZiadFrancis/Reinforcement_Trading_Part_2](https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2) — the original pipeline architecture and environment design. We trained independently from scratch and extended the pipeline with parallel training, resume support, and performance optimizations. All results below are from our own runs.

## Highlights

- **25 causal technical features** — ATR-normalized indicators, session flags, time features; no future leakage
- **Bracket-order action space** — `MultiDiscrete([direction(3), SL_buckets(3), TP_buckets(4)])` — the agent controls direction, stop-loss, and take-profit simultaneously
- **35-fold sliding walk-forward** — 5 years train → 6 months validation → 6 months test, rolling forward; stitched OOS equity curve
- **Parallel training** — CPU-only, 4 concurrent folds, full hardware utilization
- **M1 execution simulation** — intra-bar TP/SL fill logic with pessimistic SL-first assumption
- **Consistency gate** — automated pass/fail check on out-of-sample performance before deployment

## Results

### Stitched Out-of-Sample (Folds 9–35)

| Metric | Value |
|---|---|
| Total return | **+28.7%** |
| Sharpe-like ratio | **+0.22** |
| Max drawdown | **-49.0%** |
| Profit factor | **1.03** |
| Total trades | **7,434** |
| Training time | **~108 min** (28-core CPU) |

### Per-Fold Summary (27 folds)

| Fold | Return | PF | Sharpe | Max DD | Trades | Win % |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | +2.7% | 1.03 | +0.52 | -7.2% | 309 | 43% |
| 10 | -11.3% | 0.88 | -1.58 | -14.7% | 450 | 45% |
| 11 | +3.0% | 1.05 | +0.57 | -9.6% | 262 | 51% |
| 12 | +1.2% | 1.01 | +0.25 | -9.7% | 462 | 52% |
| 13 | +23.9% | 1.24 | +3.06 | -7.3% | 415 | 52% |
| 14 | **+39.0%** | 1.40 | +4.46 | -6.4% | 404 | 53% |
| 15 | -17.4% | 0.77 | -2.96 | -22.0% | 333 | 43% |
| 16 | +14.3% | 1.48 | +2.90 | -4.7% | 125 | 48% |
| 17 | -5.8% | 0.88 | -0.97 | -11.3% | 156 | 33% |
| 18 | -13.7% | 0.80 | -2.51 | -16.1% | 300 | 43% |
| 19 | +0.7% | 1.01 | +0.18 | -8.2% | 281 | 42% |
| 20 | -15.0% | 0.82 | -2.33 | -15.9% | 340 | 40% |
| 21 | -11.9% | 0.79 | -2.47 | -13.8% | 271 | 43% |
| 22 | -6.3% | 0.91 | -0.98 | -16.7% | 321 | 44% |
| 23 | +10.9% | 1.25 | +2.06 | -4.7% | 177 | 48% |
| 24 | -1.0% | 0.98 | -0.12 | -9.0% | 232 | 43% |
| 25 | -4.3% | 0.90 | -0.90 | -11.8% | 199 | 42% |
| 26 | -6.2% | 0.87 | -1.03 | -8.8% | 190 | 34% |
| 27 | +16.0% | 1.29 | +2.55 | -4.0% | 273 | 49% |
| 28 | +7.3% | 1.17 | +1.43 | -6.6% | 195 | 48% |
| 29 | -16.3% | 0.71 | -3.39 | -16.4% | 233 | 37% |
| 30 | +2.8% | 1.06 | +0.61 | -8.1% | 266 | 47% |
| 31 | +10.3% | 1.32 | +2.05 | -4.2% | 117 | 44% |
| 32 | +17.4% | 1.51 | +2.92 | -3.5% | 131 | 44% |
| 33 | -4.0% | 0.95 | -0.58 | -11.7% | 324 | 42% |
| 34 | +4.6% | 1.08 | +0.84 | -5.1% | 217 | 47% |
| 35 | +5.6% | 1.04 | +0.77 | -10.6% | 451 | 46% |

**15/27 profitable folds** (56%). Best: fold 14 (+39%). Worst: fold 15 (-17.4%).

### Consistency Gate

| Check | Result | Threshold |
|---|---|---|
| Profitable folds | 15/27 | >= 4 |
| Worst-fold PF | 0.71 | >= 0.9 |
| Mean Sharpe | +0.20 | > 0 |

**Gate result: FAILED** (worst-fold PF below threshold). Model saved to `models/best_model/` with `gate_passed=false` — not approved for deployment.

## Quick start

### Option A: Try it with sample data (no download needed)

The repo includes a 1-month data sample and a pre-trained best model (fold 14: +39% return, Sharpe 4.46) so you can explore the pipeline immediately:

```bash
pip install -r requirements.txt

# View the sample model's equity chart
open sample/ppo_H1_sl1-1.5-2_tp1-1.5-2-3_3000k_seed42_equity_insample_oos.html

# Train a single fold on the sample data
python train_ppo.py

# The sample data is at:
# sample/XAUUSD_1 Min_Bid_2025.11.csv          — Nov 2025 (27K M1 bars, ~2MB)
# sample/ppo_H1_sl1-1.5-2_tp1-1.5-2-3_3000k_seed42.zip       — trained PPO model
# sample/ppo_H1_sl1-1.5-2_tp1-1.5-2-3_3000k_seed42_vecnorm.pkl — observation normalizer
# sample/fold_summary.json                      — fold metrics (return, Sharpe, PF, etc.)
```

### Option B: Full pipeline with your own data

Place your M1 CSV in `data/`. The pipeline expects MT4/MT5-style format:

```csv
Time (EET),Open,High,Low,Close,Volume
2020.01.09 01:00:00,1557.152,1557.452,1555.202,1555.302,0.045
```

Primary dataset (long history):
```
data/XAUUSD_1 Min_Bid_2003.05.05_2026.05.31.csv   # ~23 years
```

Shorter smoke-test fallback (set `DATA_SOURCE` in `config.py`):
```
data/XAUUSD_1 Min_Ask_2020.01.09_2026.01.15.csv   # ~6 years
```

### Train the RL agent

```bash
# Parallel training (recommended — 4 concurrent folds, ~108 min on 28-core CPU)
python parallel_train.py

# Single-fold training
python train_ppo.py

# Resume from a specific fold (after interruption)
python parallel_train.py --start-fold 15

# Train specific folds only
python parallel_train.py --folds 1-5
```

### View results

```bash
# Post-hoc visualization
python view_results.py

# Interactive charts in models/sliding/fold_*/ppo_*_equity.html
```

### One-time holdout evaluation

```bash
python final_holdout_eval.py
```

## Architecture

```
XAUUSD M1 data
    │
    ▼
┌─────────────────────┐
│  data_loader.py     │  Parse CSV, resample M1→H1, walk-forward splits
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  features.py        │  25 causal indicators (RSI, ATR, BB, MACD, MFI, ...)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  env_bracket.py     │  Gymnasium env: MultiDiscrete actions, M1 execution
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  train_ppo.py       │  PPO (SB3), per-fold training, consistency gate
│  parallel_train.py  │  Parallel launcher (CPU-only, 4 concurrent)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  evaluate.py        │  Sharpe, drawdown, trade stats, stitched OOS
│  visualize.py       │  HTML equity charts
└─────────────────────┘
```

## Design decisions

- **M1 execution, H1 decisions** — the agent observes H1 bars but executes on M1 for realistic intra-bar TP/SL simulation.
- **Bracket order actions** — SL and TP are bucketed (1×, 1.5×, 2× ATR for SL; 1×, 1.5×, 2×, 3× ATR for TP). Position size is fixed-fractional; the agent only controls direction and bracket shape.
- **SL-first fill assumption** — when both TP and SL hit within the same M1 candle, the simulator assumes SL filled first (pessimistic).
- **Risk-normalized rewards** — rewards are normalized by risk budget rather than raw cash PnL for stable PPO training.
- **25 features (down from 29)** — collinearity audit (2026-06-02) dropped 4 redundant features with |r| >= 0.96.
- **No GPU used** — training is CPU-only; RTX 3090 available but not utilized.
- **5-year train / 6-month val / 6-month test** — sliding window simulates retraining every 6 months and trading the next 6 months live.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters, timeframes, risk settings |
| `data_loader.py` | CSV parsing, timezone handling, resampling, walk-forward splits |
| `features.py` | 25 causal technical indicators and stationary feature matrix |
| `env_bracket.py` | Gymnasium-compatible bracket trading environment |
| `train_ppo.py` | PPO training, walk-forward orchestration, consistency gate |
| `parallel_train.py` | Parallel multi-fold launcher (CPU-only) |
| `evaluate.py` | Performance metrics, trade summaries, drawdown analysis |
| `baselines.py` | Random, EMA-ATR, trend-hold benchmark policies |
| `visualize.py` | Plotly HTML equity charts and trade overlays |
| `view_results.py` | Post-hoc visualization viewer |
| `training_diagnostics.py` | Per-fold analysis, feature importance |
| `model_artifacts.py` | Model save/load utilities |
| `leakage_checks.py` | Data leakage validation |
| `final_holdout_eval.py` | One-time holdout evaluation entry point |
| `download_data.py` | Data download utilities |
| `run_pipeline.py` | One-command pre-test pipeline |
| `sample/` | Sample data (Nov 2025), best model (fold 14), equity chart, metrics |

## Requirements

```
pandas>=2.0
numpy>=1.24
plotly>=5.18
nbformat>=5.9
gymnasium>=0.29
stable-baselines3>=2.3
tqdm>=4.66
scikit-learn>=1.3
```

PyTorch is pulled in by `stable-baselines3` (CPU or CUDA — both work).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gold_rl_trading_2025,
  title  = {XAUUSD Reinforcement Learning Trading Pipeline},
  year   = {2025},
  url    = {https://github.com/M0-AR/gold-trading-bot}
}
```

## Contributing

Contributions welcome. Open an issue or submit a pull request.

## Disclaimer

This is a research project. The results shown are from historical backtesting and do not guarantee future performance. The out-of-sample consistency gate failed, indicating the model is not robust enough for deployment. Use at your own risk.
