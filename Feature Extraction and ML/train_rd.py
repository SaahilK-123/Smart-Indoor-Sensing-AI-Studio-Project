# train_rd.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)

from dataset_rd import build_loaders
from model_rd import RDResNet18

# ======== Config ========
CSV_PATH = "features_index.csv"
EPOCHS = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_resnet18.pt"

CURVE_PNG = "training_curves.png"
HIST_CSV = "training_history.csv"
CM_PNG = "confusion_matrix.png"
CMN_PNG = "confusion_matrix_norm.png"
CLF_TXT = "classification_report.txt"


# ======== Helpers ========
def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def _accumulate_preds(pred_list, true_list, logits, yb):
    pred_list.append(logits.argmax(1).detach().cpu().numpy())
    true_list.append(yb.detach().cpu().numpy())


@torch.no_grad()
def evaluate(model, loader: DataLoader):
    """
    Evaluate on a loader and return:
      loss_mean, acc_mean, macro_f1, weighted_f1, y_true, y_pred
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    all_pred = []
    all_true = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = F.cross_entropy(out, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out, yb) * bs
        total_n += bs

        _accumulate_preds(all_pred, all_true, out, yb)

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    loss_mean = total_loss / max(1, total_n)
    acc_mean = total_acc / max(1, total_n)
    return loss_mean, acc_mean, macro_f1, weighted_f1, y_true, y_pred


def train_one_epoch(model, loader: DataLoader, optimizer):
    """
    Train for one epoch and also compute train macro/weighted F1.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = F.cross_entropy(out, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out, yb) * bs
        total_n += bs

        _accumulate_preds(all_pred, all_true, out, yb)

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    return (
        total_loss / max(1, total_n),
        total_acc / max(1, total_n),
        macro_f1,
        weighted_f1
    )


def plot_curves(history, out_png=CURVE_PNG):
    """Plot training/validation curves: Loss, Accuracy, Macro-F1."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4.5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch");
    plt.ylabel("Loss");
    plt.title("Loss")
    plt.grid(True, alpha=0.3);
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epoch");
    plt.ylabel("Accuracy");
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3);
    plt.legend()

    # Macro-F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_f1"], label="Train F1 (macro)")
    plt.plot(epochs, history["val_f1"], label="Val F1 (macro)")
    plt.xlabel("Epoch");
    plt.ylabel("F1");
    plt.title("Macro-F1")
    plt.grid(True, alpha=0.3);
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✓ Saved curves to {out_png}")


def save_history_csv(history, out_csv=HIST_CSV):
    import pandas as pd
    df = pd.DataFrame({
        "epoch": np.arange(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "train_f1": history["train_f1"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "val_f1": history["val_f1"],
    })
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved history to {out_csv}")


def plot_confusion(y_true, y_pred, class_names, out_png_raw=CM_PNG, out_png_norm=CMN_PNG):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # raw counts
    fig, ax = plt.subplots(figsize=(6.2, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.savefig(out_png_raw, dpi=150)
    plt.close()
    print(f"✓ Saved confusion matrix to {out_png_raw}")

    # normalized by true labels
    with np.errstate(all="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    dispn = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    dispn.plot(ax=ax, cmap="Greens", colorbar=True, xticks_rotation=45, values_format=".2f")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(out_png_norm, dpi=150)
    plt.close()
    print(f"✓ Saved normalized confusion matrix to {out_png_norm}")


def save_classification_report(y_true, y_pred, class_names, out_txt=CLF_TXT):
    rpt = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, digits=4
    )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(rpt)
    print(f"✓ Saved classification report to {out_txt}")
    print("\n=== Classification Report ===\n" + rpt)


# ======== Main ========
def main():
    # === Load data & class encoder ===
    dl_train, dl_val, dl_test, le, n_classes = build_loaders(
        CSV_PATH, batch_size=BATCH_SIZE, seed=42
    )
    class_names = list(le.classes_)
    print(f"Classes ({n_classes}): {class_names}")

    # === Count samples per class (from CSV) ===
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    label_counts = df["label"].value_counts()
    print("\n=== Sample count per class ===")
    for cls_name in class_names:
        print(f"  {cls_name:<20s} : {label_counts.get(cls_name, 0)} samples")
    print("==============================\n")

    # === Model & Optimizer ===
    model = RDResNet18(n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # === Training loop (best model by validation macro-F1) ===
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_f1_w = train_one_epoch(model, dl_train, optimizer)
        va_loss, va_acc, va_f1, va_f1_w, _, _ = evaluate(model, dl_val)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        print(f"[Epoch {epoch:02d}] "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
              f"Val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({"state_dict": model.state_dict(), "classes": class_names}, CKPT_PATH)
            print(f"  ✓ Saved best (val macro-F1={va_f1:.4f}) to {CKPT_PATH}")

    # === Curves & history ===
    plot_curves(history, CURVE_PNG)
    save_history_csv(history, HIST_CSV)

    # === Test with best checkpoint ===
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print("Loaded best checkpoint by val macro-F1.")

    te_loss, te_acc, te_f1, te_f1_w, y_true, y_pred = evaluate(model, dl_test)
    print(f"[Test] loss {te_loss:.4f} | acc {te_acc:.4f} | "
          f"macro-F1 {te_f1:.4f} | weighted-F1 {te_f1_w:.4f}")

    plot_confusion(y_true, y_pred, class_names, CM_PNG, CMN_PNG)
    save_classification_report(y_true, y_pred, class_names, CLF_TXT)


if __name__ == "__main__":
    main()
