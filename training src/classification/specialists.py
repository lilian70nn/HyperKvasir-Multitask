from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from sklearn.preprocessing import LabelEncoder

from common import (
    DEVICE,
    NUM_WORKERS,
    USE_CLASS_WEIGHT,
    OUTPUT_DIR,
    make_class_weights,
    compute_metrics,
    save_eval_outputs
)

FINE_DIR = OUTPUT_DIR / "specialists"

SPECIALIST_BACKBONES = [
    {"tag": "swin_tiny", "hf_model_name": "microsoft/swin-tiny-patch4-window7-224"},
]

FINE_BATCH_SIZE = 16
FINE_EPOCHS = 20
FINE_LR = 2e-5
FINE_WEIGHT_DECAY = 1e-4
FINE_PATIENCE = 6

USE_FINE_AUGMENTATION = True

fine_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.80, 1.00)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
])



# Specialist dataset

class FineImageDataset(Dataset):
    def __init__(self, paths, labels, image_processor, label_encoder, transform=None):
        self.paths = [str(p) for p in paths]
        self.labels = [str(x) for x in labels]
        self.image_processor = image_processor
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        enc = self.image_processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        y = self.label_encoder.transform([self.labels[idx]])[0]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(y, dtype=torch.long),
            "path": self.paths[idx],
        }


# Specialist inference

@torch.no_grad()
def predict_specialist_paths(model, image_processor, label_encoder, paths, batch_size=FINE_BATCH_SIZE):
    model.eval()
    all_probs = []

    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]

        images = [Image.open(str(p)).convert("RGB") for p in batch_paths]
        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred = label_encoder.inverse_transform(pred_idx)

    return pred, conf, probs


# Train one specialist

def train_specialist(group_id, merge_label, group_classes, train_df, val_df, backbone_tag, hf_model_name):
    group_dir = FINE_DIR / f"group_{group_id:02d}" / backbone_tag
    group_dir.mkdir(parents=True, exist_ok=True)

    fine_train_df = train_df[train_df["cls_label"].isin(group_classes)].copy().reset_index(drop=True)
    fine_val_df = val_df[val_df["cls_label"].isin(group_classes)].copy().reset_index(drop=True)

    if len(fine_train_df) == 0 or len(fine_val_df) == 0:
        raise ValueError(f"Empty specialist train/val split for group {group_id}")

    y_train = fine_train_df["cls_label"].astype(str).values
    y_val = fine_val_df["cls_label"].astype(str).values
    train_paths = fine_train_df["image_path"].astype(str).values
    val_paths = fine_val_df["image_path"].astype(str).values

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    num_labels = len(label_encoder.classes_)
    id2label = {i: str(c) for i, c in enumerate(label_encoder.classes_)}
    label2id = {str(c): i for i, c in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(hf_model_name)

    model = AutoModelForImageClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    train_ds = FineImageDataset(
        paths=train_paths,
        labels=y_train,
        image_processor=image_processor,
        label_encoder=label_encoder,
        transform=fine_train_transform if USE_FINE_AUGMENTATION else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=FINE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    y_train_enc = label_encoder.transform(y_train)

    if USE_CLASS_WEIGHT:
        class_weights = make_class_weights(y_train_enc, num_labels).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_LR, weight_decay=FINE_WEIGHT_DECAY)

    best_state = None
    best_epoch = -1
    best_macro_f1 = -1.0
    patience = 0
    history = []

    print("\n" + "=" * 100)
    print("Training specialist:", merge_label)
    print("Classes:", group_classes)
    print("Backbone:", hf_model_name)
    print("Train:", len(fine_train_df), "| Val:", len(fine_val_df))
    print("=" * 100)

    for epoch in range(1, FINE_EPOCHS + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Specialist group {group_id} epoch {epoch}", leave=False):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        pred, conf, probs = predict_fine_paths(model, image_processor, label_encoder, val_paths)
        metrics = compute_metrics(y_val, pred)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        })

        print(
            f"Specialist group {group_id} epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = float(metrics["f1_macro"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= FINE_PATIENCE:
            print(f"Specialist early stopping | best_epoch={best_epoch} | best_macro_f1={best_macro_f1:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    pred, conf, probs = predict_fine_paths(model, image_processor, label_encoder, val_paths)

    eval_obj = save_eval_outputs(
        out_dir=group_dir,
        y_true=y_val,
        y_pred=pred,
        y_conf=conf,
        image_paths=val_paths,
        labels_for_cm=label_encoder.classes_.tolist(),
        prefix="specialist_",
    )

    checkpoint_path = group_dir / "specialist.pt"

    torch.save({
        "group_id": int(group_id),
        "merge_label": merge_label,
        "group_classes": list(group_classes),
        "backbone_tag": backbone_tag,
        "hf_model_name": hf_model_name,
        "model_state_dict": model.state_dict(),
        "label_classes": label_encoder.classes_.tolist(),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(group_dir / "training_history.csv", index=False)

    summary = {
        "group_id": group_id,
        "merge_label": merge_label,
        "group_classes": json.dumps(group_classes),
        "backbone_tag": backbone_tag,
        "hf_model_name": hf_model_name,
        "train_n": len(fine_train_df),
        "val_n": len(fine_val_df),
        "best_epoch": best_epoch,
        "val_accuracy": eval_obj["metrics"]["accuracy"],
        "val_balanced_accuracy": eval_obj["metrics"]["balanced_accuracy"],
        "val_f1_macro": eval_obj["metrics"]["f1_macro"],
        "checkpoint_path": str(checkpoint_path),
    }

    model.cpu()
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "summary": summary,
        "checkpoint_path": checkpoint_path,
    }


# Load specialist for final routing

def load_specialist(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    label_classes = ckpt["label_classes"]
    num_labels = len(label_classes)

    id2label = {i: str(c) for i, c in enumerate(label_classes)}
    label2id = {str(c): i for i, c in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(ckpt["hf_model_name"])

    model = AutoModelForImageClassification.from_pretrained(
        ckpt["hf_model_name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)

    return model, image_processor, label_encoder, ckpt