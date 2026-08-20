from pathlib import Path
import random
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoProcessor, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 42
NUM_WORKERS = 2
EMBED_BATCH_SIZE = 128

MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.25
MLP_BATCH_SIZE = 128
MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 15
USE_CLASS_WEIGHT = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_class_weights(y_encoded, num_classes):
    counts = np.bincount(y_encoded, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

# SigLIP embedding dataset

class SigLIPImageDataset(Dataset):
    def __init__(self, df):
        self.paths = df["image_path"].astype(str).tolist()
        self.labels = df["cls_label"].astype(str).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return image, self.labels[idx], self.paths[idx]


def siglip_collate_fn(batch):
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    paths = [x[2] for x in batch]
    return images, labels, paths



# Load one adapted SigLIP checkpoint

def load_adapted_siglip_encoder(dino_ckpt_path=None):
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        SIGLIP_MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)

    if dino_ckpt_path is None:
        dino_info = {
            "epoch": 0,
            "ssl_train_loss": None,
            "checkpoint_name": "original_siglip2",
            "checkpoint_path": None,
        }

        print("Checkpoint: original pretrained SigLIP2")
        print("DINO epoch: 0")
        print("SSL train loss: None")
        print("No DINO checkpoint loaded.")

    else:
        ckpt = torch.load(dino_ckpt_path, map_location="cpu", weights_only=False)

        if "student_state_dict" not in ckpt:
            raise KeyError(f"{dino_ckpt_path} does not contain student_state_dict")

        missing, unexpected = model.load_state_dict(
            ckpt["student_state_dict"],
            strict=False,
        )

        dino_info = {
            "epoch": int(ckpt.get("epoch", -1)),
            "ssl_train_loss": ckpt.get("ssl_train_loss"),
            "checkpoint_name": dino_ckpt_path.name,
            "checkpoint_path": str(dino_ckpt_path),
        }

        print("Checkpoint:", dino_ckpt_path.name)
        print("DINO epoch:", dino_info["epoch"])
        print("SSL train loss:", dino_info["ssl_train_loss"])
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

        del ckpt

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    gc.collect()

    return processor, model, dino_info


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_mlp(model, X_s, label_encoder, batch_size=1024):
    model.eval()
    all_probs = []

    for start in range(0, len(X_s), batch_size):
        end = min(start + batch_size, len(X_s))
        xb = torch.tensor(X_s[start:end], dtype=torch.float32, device=DEVICE)
        probs = F.softmax(model(xb), dim=1).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred = label_encoder.inverse_transform(pred_idx)

    return pred, conf, probs


def save_eval_outputs(out_dir, y_true, y_pred, y_conf, image_paths, labels_for_cm=None, prefix=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / f"{prefix}metrics.csv", index=False)

    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "label"})
    report_df.to_csv(out_dir / f"{prefix}classification_report.csv", index=False)

    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": y_true,
        "pred_label": y_pred,
        "confidence": y_conf,
        "correct": np.array(y_true) == np.array(y_pred),
    })

    pred_df.to_csv(out_dir / f"{prefix}predictions.csv", index=False)

    if labels_for_cm is None:
        labels_for_cm = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    cm_df = pd.DataFrame(cm, index=labels_for_cm, columns=labels_for_cm)
    cm_df.to_csv(out_dir / f"{prefix}confusion_matrix.csv")

    confusion_rows = []

    for i, true_label in enumerate(labels_for_cm):
        for j, pred_label in enumerate(labels_for_cm):
            if i == j:
                continue

            count = int(cm[i, j])

            if count > 0:
                confusion_rows.append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                })

    confusion_df = pd.DataFrame(confusion_rows)

    if len(confusion_df) > 0:
        confusion_df = confusion_df.sort_values("count", ascending=False).reset_index(drop=True)
    else:
        confusion_df = pd.DataFrame(columns=["true_label", "pred_label", "count"])

    confusion_df.to_csv(out_dir / f"{prefix}most_common_confusions.csv", index=False)

    wrong_df = pred_df[~pred_df["correct"]].copy().sort_values("confidence", ascending=False)
    wrong_df.to_csv(out_dir / f"{prefix}wrong_predictions_sorted_by_confidence.csv", index=False)

    return {
        "metrics": metrics,
        "metrics_df": metrics_df,
        "report_df": report_df,
        "pred_df": pred_df,
        "cm_df": cm_df,
        "confusion_df": confusion_df,
        "wrong_df": wrong_df,
    }


def make_merge_label(group):
    return "__MERGE__" + "__".join(sorted([str(x) for x in group]))