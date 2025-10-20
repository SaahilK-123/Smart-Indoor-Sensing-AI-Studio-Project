#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import csv
import json
import time
import serial
import numpy as np
from io import StringIO
import h5py

# ========================== Subcarrier Characteristics ==========================
# Feature       LLTF                         HT-LTF
# -----------------------------------------------------------------
# #Subcarriers  52 (26+26)                   112 (56+56)
# Function      Basic channel estimation     MIMO channel estimation
# Range         -26 ~ -1 and 1 ~ 26          -56 ~ -1 and 1 ~ 56
# Skipped       Guard + DC subcarriers       Guard + DC subcarriers
# Scenario      Single-user or low throughput  Multi-user or high throughput
# =============================================================================

# LLTF valid subcarriers (64-point FFT indexing example)
# -32~-27 guard, -26~-1 valid, 0 DC, 1~26 valid, 27~31 guard
csi_valid_subcarrier_index = []
CSI_VALID_SUBCARRIER_INTERVAL = 1
csi_valid_subcarrier_index += [i for i in range(6, 32, CSI_VALID_SUBCARRIER_INTERVAL)]  # first 26
csi_valid_subcarrier_index += [i for i in range(33, 59, CSI_VALID_SUBCARRIER_INTERVAL)]  # last 26
CSI_DATA_LLTF_COLUMNS = len(csi_valid_subcarrier_index)  # 52

# HT-LTF valid subcarriers (128-point FFT indexing example)
# -64~-57 guard, -56~-1 valid, 0 DC, 1~56 valid, 57~64 guard
csi_valid_subcarrier_index += [i for i in range(66, 94, CSI_VALID_SUBCARRIER_INTERVAL)]  # first 28
csi_valid_subcarrier_index += [i for i in range(95, 123, CSI_VALID_SUBCARRIER_INTERVAL)]  # last 28

# Total columns when HT-LTF available
CSI_DATA_COLUMNS = len(csi_valid_subcarrier_index)  # 52 + 56 = 108

# CSV column names (metadata + CSI data)
DATA_COLUMNS_NAMES = [
    "type",  # 0: Packet type (e.g., management, control, data frame)
    "id",  # 1: Unique packet identifier
    "mac",  # 2: Sender MAC address
    "rssi",  # 3: Received Signal Strength Indicator
    "rate",  # 4: Data transmission rate (Mbps)
    "sig_mode",  # 5: Signal mode (0: legacy, 1: HT)
    "mcs",  # 6: Modulation and Coding Scheme
    "bandwidth",  # 7: Bandwidth: 0=20 MHz, 1=40 MHz, 2=80 MHz, 3=160 MHz
    "smoothing",  # 8: Channel smoothing enabled (1=yes, 0=no)
    "not_sounding",  # 9: Sounding disabled (0=enabled, 1=disabled)
    "aggregation",  # 10: Frame aggregation (1=yes, 0=no)
    "stbc",  # 11: Space-time block coding (1=enabled, 0=disabled)
    "fec_coding",  # 12: Forward error correction (1=enabled, 0=disabled)
    "sgi",  # 13: Short Guard Interval (1=enabled, 0=disabled)
    "noise_floor",  # 14: Noise floor (dBm)
    "ampdu_cnt",  # 15: Aggregated MPDU count
    "channel",  # 16: Wi-Fi primary channel number
    "secondary_channel",  # 17: Secondary channel position (-1=left, 0=none, 1=right)
    "local_timestamp",  # 18: Local device timestamp (microseconds since boot)
    "ant",  # 19: Antenna index
    "sig_len",  # 20: Signal length (bytes)
    "rx_state",  # 21: Reception status (success/fail)
    "len",  # 22: Total packet length (bytes)
    "first_word",  # 23: First word (decoder-related)
    "data"  # 24: CSI raw data: [imag_1, real_1, imag_2, real_2, ...]
]

# Valid CSI raw array lengths observed in ESP32 output
VALID_CSI_LENGTHS = {128, 256, 384}


def open_serial(port: str, baudrate: int = 921600, timeout: float = 1.0):
    """Open serial port with given settings."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=timeout
        )
        if not ser.isOpen():
            print("open failed")
            return None
        print("open success")
        return ser
    except Exception as e:
        print(f"open failed: {e}")
        return None


def parse_csi_row(row: list):
    """
    Parse one CSV row into a fixed-length CSI vector and metadata.
    - Returns CSI vector of length CSI_DATA_COLUMNS (complex64).
      If only LLTF (n==128), pad the tail with NaN+0j to match length.
    - local_timestamp is mapped to local wall-clock (YYYY-MM-DD HH:MM:SS.mmm).
    """
    expected_cols = len(DATA_COLUMNS_NAMES)
    if len(row) != expected_cols:
        return None, None

    # Parse raw CSI JSON array (interleaved [imag, real])
    try:
        csi_raw = json.loads(row[-1])
    except json.JSONDecodeError:
        return None, None

    n = len(csi_raw)
    if n not in VALID_CSI_LENGTHS:
        return None, None

    # Determine how many valid subcarriers to extract
    use_len = CSI_DATA_LLTF_COLUMNS if n == 128 else CSI_DATA_COLUMNS
    # Build fixed-length complex vector (pad tail with NaN+0j if LLTF)
    csi_fixed = np.full((CSI_DATA_COLUMNS,), np.nan + 0j, dtype=np.complex64)
    try:
        for i in range(use_len):
            re = csi_raw[csi_valid_subcarrier_index[i] * 2 + 1]
            im = csi_raw[csi_valid_subcarrier_index[i] * 2]
            csi_fixed[i] = complex(re, im)
    except (IndexError, TypeError):
        return None, None

    # Convert ESP32 microsecond timestamp to local human-readable time (.mmm)
    try:
        esp32_us = int(row[18])
    except Exception:
        return None, None

    if not hasattr(parse_csi_row, "_t0_epoch"):
        parse_csi_row._t0_epoch = time.time()
        parse_csi_row._t0_esp32_us = esp32_us
    if esp32_us < parse_csi_row._t0_esp32_us:
        parse_csi_row._t0_epoch = time.time()
        parse_csi_row._t0_esp32_us = esp32_us

    dt_s = (esp32_us - parse_csi_row._t0_esp32_us) / 1e6
    local_epoch_s = parse_csi_row._t0_epoch + dt_s
    tm = time.localtime(local_epoch_s)
    ms = int((local_epoch_s - int(local_epoch_s)) * 1000)
    local_str = time.strftime("%Y-%m-%d %H:%M:%S", tm) + f".{ms:03d}"

    # Metadata (focus on varying and useful fields)
    meta = {
        # strongly varying
        'rssi': row[3],
        'noise_floor': row[14],
        'channel': row[16],
        'secondary_channel': row[17],
        'local_timestamp': local_str,
        'len': row[22],
        # possibly fixed but useful
        'rate': row[4],
        'sig_mode': row[5],
        'mcs': row[6],
        'bandwidth': row[7],
        'sgi': row[13],
    }
    return csi_fixed, meta


def main():
    # Serial port device path (modify if needed)
    #  ls /dev/cu.* for macos
    # COMX for windows (where X is an int)
    # serial_port = '/dev/cu.usbserial-0001'
    serial_port = 'COM6'

    ser = open_serial(serial_port)
    if ser is None:
        return

    # --- Open HDF5 file for writing (append mode) ---
    h5_filename = time.strftime("csi_%Y-%m-%d_%H-%M-%S.h5", time.localtime())
    f = h5py.File(h5_filename, "a")
    print(f"[HDF5] Creating file: {h5_filename}")

    # Create expandable datasets if absent
    if "csi" not in f:
        f.create_dataset(
            "csi",
            shape=(0, CSI_DATA_COLUMNS),
            maxshape=(None, CSI_DATA_COLUMNS),
            dtype=np.complex64,
            chunks=True
        )
    if "meta" not in f:
        f.create_dataset(
            "meta",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True
        )

    dset_csi = f["csi"]
    dset_meta = f["meta"]

    buffer_csi = []
    buffer_meta = []
    BATCH_SIZE = 50  # smaller to see writes sooner
    total_saved = 0

    try:
        while True:
            line = ser.readline()
            if not line:
                continue

            s = line.decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" not in s:
                continue

            try:
                row = next(csv.reader(StringIO(s)))
            except Exception:
                continue

            csi_vec, meta = parse_csi_row(row)
            if csi_vec is None:
                continue

            # Per-frame print
            print(f"[Frame] time={meta['local_timestamp']} rssi={meta['rssi']} dBm "
                  f"ch={meta['channel']} len={meta['len']} vecN={csi_vec.shape[0]}")

            buffer_csi.append(csi_vec)  # shape (108,)
            buffer_meta.append(json.dumps(meta))  # store meta as JSON string

            # Batch write
            if len(buffer_csi) >= BATCH_SIZE:
                try:
                    old_len = dset_csi.shape[0]
                    new_len = old_len + len(buffer_csi)

                    dset_csi.resize(new_len, axis=0)
                    dset_meta.resize(new_len, axis=0)

                    # stack to (batch, 108)
                    dset_csi[old_len:new_len] = np.vstack(buffer_csi).astype(np.complex64)
                    dset_meta[old_len:new_len] = buffer_meta

                    total_saved += len(buffer_csi)
                    print(f"[HDF5] Saved {len(buffer_csi)} frames, total={total_saved}")

                    buffer_csi.clear()
                    buffer_meta.clear()
                    f.flush()
                except Exception as e:
                    print(f"[HDF5][ERROR] {type(e).__name__}: {e}")
                    buffer_csi.clear()
                    buffer_meta.clear()

    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...")

    finally:
        # Final flush remaining buffers
        try:
            if buffer_csi:
                old_len = dset_csi.shape[0]
                new_len = old_len + len(buffer_csi)
                dset_csi.resize(new_len, axis=0)
                dset_meta.resize(new_len, axis=0)
                dset_csi[old_len:new_len] = np.vstack(buffer_csi).astype(np.complex64)
                dset_meta[old_len:new_len] = buffer_meta
                total_saved += len(buffer_csi)
                f.flush()
                print(f"[HDF5] Final flush {len(buffer_csi)} frames, total={total_saved}")
        except Exception as e:
            print(f"[HDF5][FINAL][ERROR] {type(e).__name__}: {e}")
        finally:
            try:
                ser.close()
            except Exception:
                pass
            f.close()
            print(f"[Done] HDF5 file closed, total {total_saved} frames saved to {h5_filename}")


if __name__ == '__main__':
    # Optional: version check
    if sys.version_info < (3, 6):
        print("Python version should >= 3.6")
        sys.exit(1)
    main()
