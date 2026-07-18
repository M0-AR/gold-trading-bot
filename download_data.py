#!/usr/bin/env python3
"""
Download all XAUUSD M1 data from Dukascopy (2003-2026).
Stores raw chunks + merged CSV inside the project data/ directory.
Usage: python download_data.py
"""
import subprocess, os, sys, time, glob, csv
from datetime import datetime, timezone, timedelta
import calendar

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHUNK_DIR = os.path.join(DATA_DIR, "dukascopy_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

# --- Dukascopy year ranges ---
RANGES = []
RANGES.append(("2003-05-05", "2004-01-01"))
for y in range(2004, 2026):
    RANGES.append((f"{y}-01-01", f"{y+1}-01-01"))
RANGES.append(("2026-01-01", "2026-06-01"))


def download_range(fr, to, instrument="xauusd", price_type="bid"):
    """Download a date range from Dukascopy via dukascopy-cli."""
    # dukascopy-cli outputs: {instrument}-m1-{price_type}-{from}-{to}.csv
    outfile = os.path.join(CHUNK_DIR, f"{instrument}-m1-{price_type}-{fr}-{to}.csv")
    if os.path.exists(outfile) and os.path.getsize(outfile) > 1000:
        with open(outfile) as f:
            lines = sum(1 for _ in f) - 1
        print(f"  SKIP (exists, {lines} lines): {fr} -> {to}")
        return outfile
    # Use full path to npx — subprocess may not inherit shell PATH
    import shutil
    npx_path = shutil.which("npx") or "/home/md/.nvm/versions/node/v24.15.0/bin/npx"
    cmd = [
        npx_path, "dukascopy-cli",
        "-i", instrument,
        "-from", fr, "-to", to,
        "-t", "m1", "-p", price_type, "-v", "-f", "csv",
        "-dir", CHUNK_DIR, "-bs", "20", "-bp", "300",
        "-r", "3", "-rp", "2000", "-s"
    ]
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(npx_path) + ":" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if os.path.exists(outfile):
        with open(outfile) as f:
            lines = sum(1 for _ in f) - 1
        print(f"  OK {fr} -> {to}: {lines} lines")
    else:
        print(f"  FAIL {fr} -> {to}")
        if result.stderr:
            print(f"    {result.stderr[:200]}")
    return outfile


def is_eest(dt):
    """Check if datetime is in EEST (Eastern European Summer Time)."""
    y = dt.year
    cal = calendar.Calendar()
    mar = [d for d in cal.itermonthdates(y, 3) if d.month == 3 and d.weekday() == 6]
    oct_ = [d for d in cal.itermonthdates(y, 10) if d.month == 10 and d.weekday() == 6]
    if not mar or not oct_:
        return False
    return mar[-1] <= dt.date() < oct_[-1]


def utc_ms_to_eet_str(utc_ms):
    """Convert UTC milliseconds to EET string 'YYYY.MM.DD HH:MM:SS'."""
    dt_utc = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc)
    off = timedelta(hours=3) if is_eest(dt_utc) else timedelta(hours=2)
    return (dt_utc + off).strftime("%Y.%m.%d %H:%M:%S")


def merge_and_convert(chunks, out_path, price_type="bid"):
    """Merge chunk CSVs, convert timestamps from UTC ms to EET, dedup, sort."""
    all_rows = []
    for fpath in chunks:
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 100:
            continue
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(row['timestamp'])
                    o = float(row['open'])
                    h = float(row['high'])
                    l = float(row['low'])
                    c = float(row['close'])
                    v = float(row['volume'])
                    all_rows.append((ts, o, h, l, c, v))
                except (ValueError, KeyError):
                    continue

    print(f"Total raw rows: {len(all_rows)}")

    # Dedup by timestamp
    seen = set()
    unique = []
    for row in all_rows:
        if row[0] not in seen:
            seen.add(row[0])
            unique.append(row)
    print(f"After dedup: {len(unique)}")

    # Sort by timestamp
    unique.sort(key=lambda x: x[0])

    # Convert and write
    with open(out_path, 'w') as f:
        f.write("Time (EET),Open,High,Low,Close,Volume\n")
        for ts, o, h, l, c, v in unique:
            ts_str = utc_ms_to_eet_str(ts)
            f.write(f"{ts_str},{o},{h},{l},{c},{v}\n")
    print(f"Written {len(unique)} rows to {out_path}")


def main():
    print(f"=== XAUUSD M1 Data Download (Dukascopy) ===")
    print(f"Ranges: {len(RANGES)}, output: {DATA_DIR}/")

    # 1) Download Bid data
    print(f"\n--- Downloading BID data ---")
    bid_chunks = []
    for i, (fr, to) in enumerate(RANGES):
        print(f"[{i+1}/{len(RANGES)}]", end="")
        path = download_range(fr, to, "xauusd", "bid")
        bid_chunks.append(path)

    bid_out = os.path.join(DATA_DIR, "XAUUSD_1 Min_Bid_2003.05.05_2026.05.31.csv")
    print(f"\n--- Converting BID data ---")
    merge_and_convert(bid_chunks, bid_out, "bid")

    # 2) Download Ask data (2020-01-09 to 2026-01-15 per config)
    print(f"\n--- Downloading ASK data ---")
    ask_ranges = [
        ("2020-01-09", "2020-07-01"),
        ("2020-07-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", "2026-01-16"),
    ]
    ask_chunks = []
    for i, (fr, to) in enumerate(ask_ranges):
        print(f"[{i+1}/{len(ask_ranges)}]", end="")
        path = download_range(fr, to, "xauusd", "ask")
        ask_chunks.append(path)

    ask_out = os.path.join(DATA_DIR, "XAUUSD_1 Min_Ask_2020.01.09_2026.01.15.csv")
    print(f"\n--- Converting ASK data ---")
    merge_and_convert(ask_chunks, ask_out, "ask")

    # Summary
    print(f"\n=== DONE ===")
    for f in [bid_out, ask_out]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            with open(f) as fh:
                lines = sum(1 for _ in fh) - 1
            print(f"  {os.path.basename(f)}: {lines:,} bars, {sz/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
