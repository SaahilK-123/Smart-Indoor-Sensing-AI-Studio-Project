# dataset_rd.py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class RDDataset(Dataset):
    def __init__(self, rows, label_encoder, target_t=256, target_r=128):
        self.rows = rows
        self.le = label_encoder
        self.target_t = target_t
        self.target_r = target_r

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fp = self.rows[idx]["filepath"]
        y_str = self.rows[idx]["label"]
        y = int(self.le.transform([y_str])[0])

        x = np.load(fp)
        if x.ndim != 2:
            raise RuntimeError(f"Expected 2D (T,R), got {x.shape}")

        # log compress + resize to (target_t, target_r) + per-sample z-norm
        x = np.log1p(x)
        xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # [1,1,T,R]
        xt = F.interpolate(xt, size=(self.target_t, self.target_r),
                           mode="bilinear", align_corners=False).squeeze(0)  # [1,T,R]
        xt = (xt - xt.mean()) / (xt.std() + 1e-6)
        return xt, y


def build_loaders(csv_path, batch_size=32, seed=42):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.fit(df["label"])
    n_classes = len(le.classes_)

    # simple random split 70/15/15
    train, val, test = np.split(
        df.sample(frac=1, random_state=seed),
        [int(.7 * len(df)), int(.85 * len(df))]
    )

    def make_ds(sub): return RDDataset(sub.to_dict("records"), le)

    dl_train = DataLoader(make_ds(train), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(make_ds(val), batch_size=batch_size)
    dl_test = DataLoader(make_ds(test), batch_size=batch_size)
    return dl_train, dl_val, dl_test, le, n_classes
