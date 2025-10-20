# -*- coding: utf-8 -*-
"""
Process a list of CSI .h5 files:
- Read label intervals from a multi-sheet Excel (Date, Interval Start/End Time, Environment Classification)
- Sliding window on CSI, DC filled by mean(±1), power -> time-uniformization -> range zoom-FFT -> Doppler zoom-FFT
- Feature = abs(Doppler-Range), saved as .npy under Feature/
- Realtime append to CSV "features_index.csv": one line per window (filepath,label)
- Filenames include: <h5_stem>_start<start>_end<end>_label_<Label>.npy

Only two functions are kept: `dbinv` and `get_scaled_csi_esp32`.
"""

import os
import re
import json
import h5py
import datetime as dt
import numpy as np
import pandas as pd
from scipy.signal import zoom_fft
from pathlib import Path

# ======================== Constants ========================
C = 299_792_458.0  # m/s
DELTA_F = 312_500.0  # Hz (802.11n, 20 MHz)
RANGE_BINS = 128
RANGE_MIN_M, RANGE_MAX_M = 0.0, 32.0
DOPPLER_BINS = 256
F_D_MIN_HZ, F_D_MAX_HZ = -150.0, 150.0
win, inc = 256, 128  # sliding window (frames)


# ======================== Kept functions only ========================
def dbinv(db: float) -> float:
    """Convert dB to linear power."""
    return 10.0 ** (db / 10.0)


def get_scaled_csi_esp32(csi_1d, rssi_dbm, noise_dbm=None, use_noise=False):
    """
    Scale one CSI vector by RSSI (optionally subtracting noise power).
    Returns complex128 CSI scaled to a consistent power level.
    """
    csi = np.asarray(csi_1d, dtype=np.complex128)
    csi_pwr = np.mean(np.abs(csi) ** 2) + 1e-12
    rssi_pwr = dbinv(float(rssi_dbm))
    scale = rssi_pwr / csi_pwr
    if use_noise and (noise_dbm is not None):
        noise_pwr = dbinv(float(noise_dbm))
        eff_pwr = max(rssi_pwr - noise_pwr, 1e-12)
        scale = eff_pwr / csi_pwr
    return (csi * np.sqrt(scale) * 1e3).astype(np.complex128)


# ======================== Paths & H5 file list ========================
# Root directory containing CSI .h5 files
H5_PATH = "./"

# Explicit list of .h5 filenames you want to process (from your screenshot)
H5_FILES = [
    "csi_2025-09-23_14-34-21.h5",
    "csi_2025-09-23_14-34-32.h5",
    "csi_2025-09-30_17-05-02.h5",
    "csi_2025-09-30_17-05-04.h5",
    "csi_2025-09-30_18-04-58.h5",
    "csi_2025-09-30_18-05-01.h5",
    "csi_2025-09-30_18-05-02.h5",
]

# Combine into full absolute paths
H5_LIST = [str(Path(H5_PATH) / f) for f in H5_FILES]

# Excel with labels
EXCEL_PATH = "./Application Studio A _ 2025 IOT Sensoring.xlsx"

# Outputs
FEATURE_DIR = Path("Feature");
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
CSV_INDEX_PATH = Path("features_index.csv")
CSV_HEADER = "filepath,label\n"
if not CSV_INDEX_PATH.exists():
    with open(CSV_INDEX_PATH, "w", encoding="utf-8-sig", newline="") as f:
        f.write(CSV_HEADER)

print(f"Found {len(H5_LIST)} CSI H5 files:")
for p in H5_LIST: print("  ", p)

# ======================== Read Excel (merge all sheets) ========================
excel_path = Path(EXCEL_PATH)
if not excel_path.exists():
    raise FileNotFoundError(f"Excel not found: {excel_path}")

xls = pd.ExcelFile(excel_path)
all_sched = []
for sh in xls.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sh, dtype=str)
    if df.empty:
        continue
    df.columns = [c.strip() for c in df.columns]


    # Fuzzy column finder (case-insensitive "contains")
    def _find_col(keyword_like: str) -> str:
        kw = keyword_like.lower()
        for c in df.columns:
            if kw in c.lower():
                return c
        raise KeyError(f"[{sh}] Column containing '{keyword_like}' not found. "
                       f"Available: {list(df.columns)}")


    date_col = _find_col("date")
    start_col = _find_col("interval start")
    end_col = _find_col("interval end")
    label_col = _find_col("environment classification")

    # Parse date (often dd/mm/yyyy) and times HH:MM:SS
    date_s = pd.to_datetime(df[date_col].astype(str).str.strip(), dayfirst=True, errors="coerce")
    start_t = pd.to_datetime(df[start_col].astype(str).str.strip(), format="%H:%M:%S", errors="coerce").dt.time
    end_t = pd.to_datetime(df[end_col].astype(str).str.strip(), format="%H:%M:%S", errors="coerce").dt.time

    sched = pd.DataFrame({
        "start_dt": [pd.Timestamp.combine(d.date(), s) if pd.notna(d) and pd.notna(s) else pd.NaT
                     for d, s in zip(date_s, start_t)],
        "end_dt": [pd.Timestamp.combine(d.date(), e) if pd.notna(d) and pd.notna(e) else pd.NaT
                   for d, e in zip(date_s, end_t)],
        "label": df[label_col].astype(str).str.strip()
    }).dropna(subset=["start_dt", "end_dt"])

    # Handle cross-midnight intervals: if end < start, push end to next day
    cross = sched["end_dt"] < sched["start_dt"]
    sched.loc[cross, "end_dt"] = sched.loc[cross, "end_dt"] + pd.Timedelta(days=1)

    all_sched.append(sched)

if not all_sched:
    raise ValueError("No valid intervals parsed from any sheet.")

# Merge all sheets, drop duplicates, sort by start time
sched_all = pd.concat(all_sched, ignore_index=True)
sched_all = sched_all.drop_duplicates(subset=["start_dt", "end_dt", "label"])
sched_all = sched_all.sort_values("start_dt").reset_index(drop=True)

# Build interval index (left-closed, right-open) + label array
iv = pd.IntervalIndex.from_arrays(sched_all["start_dt"], sched_all["end_dt"], closed="left")
labs = sched_all["label"].reset_index(drop=True)

print(f"Loaded {len(sched_all)} labeled intervals from Excel.")

# ======================== Process each H5 file ========================
for file_idx, h5_file in enumerate(H5_LIST, 1):
    h5_path = Path(h5_file)
    if not h5_path.exists():
        print(f"[{file_idx}/{len(H5_LIST)}] WARNING: H5 not found, skip -> {h5_file}")
        continue

    print(f"\n========== [{file_idx}/{len(H5_LIST)}] Processing {h5_path.name} ==========")

    with h5py.File(h5_path, "r", locking=False) as f:
        dset_csi = f["csi"]  # (N,108) complex64
        dset_meta = f["meta"]  # (N,)   json strings
        N = dset_csi.shape[0]

        # Parse metadata: local_timestamp and RSSI per frame
        meta = [json.loads(m) for m in dset_meta[:]]
        times_local = np.array([
            dt.datetime.strptime(m["local_timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            for m in meta
        ])
        rssi_arr = np.array([float(m.get("rssi", -50)) for m in meta], dtype=float)

        starts = list(range(0, max(0, N - win + 1), inc))
        total_windows = len(starts)
        print(f"  Total windows in file: {total_windows}")

        for i, start in enumerate(starts, 1):
            end = start + win

            # Current window timestamps
            t_dt = times_local[start:end]

            # (1) Take first 52 LLTF subcarriers (exclude DC)
            csi52 = dset_csi[start:end, :52].astype(np.complex128)

            # (2) Per-frame RSSI scaling
            for k in range(csi52.shape[0]):
                csi52[k, :] = get_scaled_csi_esp32(csi52[k, :], rssi_dbm=rssi_arr[start + k])

            # (3) Fill DC subcarrier by averaging ±1  → 53 subcarriers (-26..-1, 0, +1..+26)
            csi_interp = np.zeros((csi52.shape[0], 53), dtype=np.complex128)
            csi_interp[:, :26] = csi52[:, :26]  # negative bins -26..-1
            csi_interp[:, 27:] = csi52[:, 26:]  # positive bins +1..+26
            csi_interp[:, 26] = 0.5 * (csi52[:, 25] + csi52[:, 26])  # DC = mean(-1, +1)

            # (4) Power per subcarrier
            CSI_PWR = np.real(csi_interp * np.conj(csi_interp))  # (T, 53)

            # (5) Time uniformization (required for Doppler FFT)
            #     Build relative time in seconds w.r.t the first frame in the window.
            t_rel = np.array([(t - t_dt[0]).total_seconds() for t in t_dt], dtype=float)
            keep = np.r_[True, np.diff(t_rel) > 0]  # drop non-increasing stamps
            t_rel = t_rel[keep]
            CSI_PWR = CSI_PWR[keep, :]

            dur = t_rel[-1] - t_rel[0]
            if dur <= 0:
                print(f"    [{i}/{total_windows}] start={start} skipped (degenerate time axis).")
                continue

            # Estimate sampling rate and build uniform time vector
            fs_est = (len(t_rel) - 1) / dur
            t_u = np.linspace(t_rel[0], t_rel[-1], int(np.floor(dur * fs_est)) + 1)

            # Interpolate each subcarrier column to uniform time; column-wise mean removal
            CSI_PWR_u = np.array([np.interp(t_u, t_rel, col) for col in CSI_PWR.T]).T
            CSI_PWR_u -= np.mean(CSI_PWR_u, axis=0, keepdims=True)

            # (6) Range zoom-FFT along frequency/subcarrier axis (axis=1)
            range_fft = zoom_fft(
                CSI_PWR_u,
                fn=[RANGE_MIN_M, RANGE_MAX_M],
                m=RANGE_BINS,
                fs=C / DELTA_F,
                axis=1
            )

            # (7) Doppler zoom-FFT along time axis (axis=0), using estimated fs
            doppler_fft = zoom_fft(
                range_fft,
                fn=[F_D_MIN_HZ, F_D_MAX_HZ],
                m=DOPPLER_BINS,
                fs=fs_est,
                axis=0
            )

            # (8) Label the window by majority vote; fallback to center time if needed
            idx_in_iv = iv.get_indexer(pd.Series(t_dt))
            labels_series = pd.Series([labs[j] if j != -1 else None for j in idx_in_iv])
            if labels_series.dropna().empty:
                center_idx = len(t_dt) // 2
                j = iv.get_indexer(pd.Series([t_dt[center_idx]]))[0]
                label_major = labs[j] if j != -1 else "Unknown"
            else:
                label_major = labels_series.value_counts(dropna=True).idxmax()

            # (9) Feature = magnitude of Doppler-Range response
            feat = np.abs(doppler_fft).astype(np.float32)

            # (10) Save .npy feature; include h5 stem + window + label
            safe_label = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]+", "_", str(label_major))
            base = h5_path.stem
            fn = f"{base}_start{start}_end{end}_label_{safe_label}.npy"
            out_path = FEATURE_DIR / fn
            np.save(out_path, feat)

            # (11) REAL-TIME append to CSV (filepath,label), flush immediately
            with open(CSV_INDEX_PATH, "a", encoding="utf-8-sig", newline="") as fcsv:
                fcsv.write(f"{out_path.resolve()},{label_major}\n")
                fcsv.flush()
                os.fsync(fcsv.fileno())

            # (12) Progress print
            print(f"    [{i}/{total_windows}] {base}  start={start}, end={end}, "
                  f"frames={end - start}, fs={fs_est:.2f} Hz, label={label_major}, "
                  f"saved -> {out_path.name}")

print("\n✅ Done. Features saved under:", FEATURE_DIR.resolve())
print("✅ Real-time index CSV:", CSV_INDEX_PATH.resolve())
