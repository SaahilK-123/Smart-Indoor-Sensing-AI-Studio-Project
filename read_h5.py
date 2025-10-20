import h5py, json
import numpy as np


path = "csi_2025-09-30_17-05-04.h5"

with h5py.File(path, "r") as f:
    # 1) Check if file is valid and open
    print("file id valid:", f.id.valid)  # Should be True
    print("top-level keys:", list(f.keys()))  # Should contain ['csi', 'meta']

    # 2) Access datasets
    dset_csi = f["csi"]
    dset_meta = f["meta"]

    print("csi shape:", dset_csi.shape, "dtype:", dset_csi.dtype)  # (N, 108) complex64
    print("meta shape:", dset_meta.shape, "dtype:", dset_meta.dtype)  # (N,) string

    # 3) Read data selectively (avoid loading the whole file into memory)
    # Example: read the first 10 frames
    csi_10 = dset_csi[0:10]  # shape (10, 108)
    meta_raw = dset_meta[0:10]  # shape (10,)

    # Convert the first meta entry (JSON string) back to a Python dictionary
    meta_0 = json.loads(meta_raw[0])
    print("first meta:", meta_0)

    # Print the first CSI vector (complex values)
    print("first CSI vector:", csi_10[0])

