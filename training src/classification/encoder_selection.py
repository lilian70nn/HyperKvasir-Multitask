"""
Encoder selection and merge-group discovery.

This module:
1. Builds a global stratified 80/20 train/validation split.
2. Evaluates the original pretrained SigLIP2 encoder and each DINO-adapted
   checkpoint using an MLP probe classifier.
3. Selects the encoder with the highest validation macro-F1.
4. Uses the winning probe's confusion matrix to identify strongly confused
   leaf classes and construct merge groups.
5. Saves the selected encoder information, data split, merge groups, and
   leaf-to-coarse mapping for the hierarchical classifier.

The probe classifier is used only for encoder selection and merge-group
discovery. 
"""


from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from IPython.display import display

from .config import *
from .common import *


def clean_text(x):
    x = str(x).strip().lower()
    x = x.replace("\\", "/")
    x = x.replace("_", "-")
    x = x.replace(" ", "-")
    return "-".join([p for p in x.split("-") if p])


def build_labeled_image_index():
    image_index = {}
    for p in LABELED_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_index[p.stem] = str(p)
    print("Indexed labeled images:", len(image_index))
    if len(image_index) == 0:
        raise ValueError(f"No images found under {LABELED_ROOT}")
    return image_index



def load_count_gt_100_leaf_df():
    df = pd.read_csv(LABEL_CSV)

    required_cols = ["Video file", "Organ", "Classification", "Finding"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["video_file"] = df["Video file"].astype(str).str.strip()
    df["Organ"] = df["Organ"].apply(clean_text)
    df["Classification"] = df["Classification"].apply(clean_text)
    df["Finding"] = df["Finding"].apply(clean_text)
    df["full_label_reference"] = df["Organ"] + "/" + df["Classification"] + "/" + df["Finding"]
    df["cls_label"] = df["Finding"]

    image_index = build_labeled_image_index()
    df["image_path"] = df["video_file"].apply(lambda x: image_index.get(Path(x).stem))
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    counts = df["cls_label"].value_counts()
    keep_classes = sorted(counts[counts > MIN_CLASS_COUNT].index.tolist())
    df = df[df["cls_label"].isin(keep_classes)].reset_index(drop=True)

    count_df = df["cls_label"].value_counts().rename_axis("cls_label").reset_index(name="count").sort_values("count", ascending=False)

    print("\n========== Count > 100 leaf data ==========")
    print("Total images:", len(df))
    print("Num classes:", len(keep_classes))
    display(count_df)

    df.to_csv(OUTPUT_DIR / "all_count_gt_100_leaf_images.csv", index=False)
    count_df.to_csv(OUTPUT_DIR / "count_gt_100_leaf_class_counts.csv", index=False)

    return df, keep_classes


def make_global_split():
    df, leaf_classes = load_count_gt_100_leaf_df()

    train_df, val_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=df["cls_label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    split_counts = pd.concat([
        train_df["cls_label"].value_counts().rename("train_count"),
        val_df["cls_label"].value_counts().rename("val_count"),
    ], axis=1).fillna(0).astype(int)

    split_counts["total_count"] = split_counts["train_count"] + split_counts["val_count"]
    split_counts = split_counts.loc[leaf_classes]

    train_df.to_csv(OUTPUT_DIR / "global_train_df.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "global_val_df.csv", index=False)
    split_counts.to_csv(OUTPUT_DIR / "global_split_counts.csv")

    print("\n========== Global 80/20 split ==========")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    display(split_counts)

    return train_df, val_df, leaf_classes, split_counts


@torch.no_grad()
def extract_siglip_embeddings_no_save(df, processor, model, split_name):
    ds = SigLIPImageDataset(df)

    loader = DataLoader(
        ds,
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        collate_fn=siglip_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    all_embs = []

    for images, _, _ in tqdm(loader, desc=f"Extracting {split_name} embeddings"):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        emb = outputs.pooler_output

        if emb is None:
            raise RuntimeError("SigLIP2 did not return pooler_output")

        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0).numpy().astype(np.float32)


def train_probe_classifier(X_train_s, y_train, X_val_s, y_val):
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    y_train_enc = label_encoder.transform(y_train)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = MLPHead(
        input_dim=X_train_s.shape[1],
        hidden_dim=MLP_HIDDEN_DIM,
        num_classes=len(label_encoder.classes_),
        dropout=MLP_DROPOUT,
    ).to(DEVICE)

    if USE_CLASS_WEIGHT:
        class_weights = make_class_weights(y_train_enc, len(label_encoder.classes_)).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    best_state = None
    best_epoch = -1
    best_macro_f1 = -1.0
    best_pred = None
    best_conf = None
    patience = 0

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        pred, conf, _ = predict_mlp(model, X_val_s, label_encoder)
        metrics = compute_metrics(y_val, pred)

        print(
            f"Probe epoch {epoch:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = float(metrics["f1_macro"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = np.array(pred).astype(str)
            best_conf = np.array(conf).astype(float)
            patience = 0
        else:
            patience += 1

        if patience >= MLP_PATIENCE:
            print(f"Probe early stopping | best_epoch={best_epoch} | best_macro_f1={best_macro_f1:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "label_encoder": label_encoder,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "pred": best_pred,
        "conf": best_conf,
    }


def find_connected_components(nodes, edges):
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups = {}

    for x in nodes:
        root = find(x)
        groups.setdefault(root, []).append(x)

    components = [sorted(group) for group in groups.values() if len(group) >= 2]
    return sorted(components, key=lambda x: (len(x), x), reverse=True)



def discover_merge_groups(y_true, y_pred, leaf_classes):
    labels = list(leaf_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    pair_rows = []
    edges = []

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j:
                continue

            a_to_b = int(cm[i, j])
            b_to_a = int(cm[j, i])
            mutual = a_to_b + b_to_a

            pair_rows.append({
                "class_a": a,
                "class_b": b,
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "mutual_confusion": mutual,
                "merge": mutual >= MERGE_THRESHOLD,
            })

            if mutual >= MERGE_THRESHOLD:
                edges.append((a, b))

    pair_df = pd.DataFrame(pair_rows).sort_values("mutual_confusion", ascending=False).reset_index(drop=True)
    merge_groups = find_connected_components(labels, edges)

    leaf_to_coarse = {leaf: leaf for leaf in labels}

    group_rows = []

    for group_id, group in enumerate(merge_groups, start=1):
        merge_label = make_merge_label(group)

        for leaf in group:
            leaf_to_coarse[leaf] = merge_label

        group_rows.append({
            "merge_group_id": group_id,
            "merge_label": merge_label,
            "num_classes": len(group),
            "classes": json.dumps(group),
        })

    merge_group_df = pd.DataFrame(group_rows)

    cm_df.to_csv(ENCODER_SELECTION_DIR / "winning_probe_confusion_matrix.csv")
    pair_df.to_csv(ENCODER_SELECTION_DIR / "winning_probe_confusion_pairs.csv", index=False)
    merge_group_df.to_csv(ENCODER_SELECTION_DIR / "merge_groups.csv", index=False)

    mapping_df = pd.DataFrame([
        {"leaf_label": leaf, "coarse_label": coarse, "is_merged": leaf != coarse}
        for leaf, coarse in leaf_to_coarse.items()
    ])

    mapping_df.to_csv(ENCODER_SELECTION_DIR / "leaf_to_coarse_mapping.csv", index=False)

    print("\n========== Winning probe merge discovery ==========")
    print("MERGE_THRESHOLD:", MERGE_THRESHOLD)
    print("\nTop confusion pairs:")
    display(pair_df.head(30))
    print("\nMerge groups:")
    display(merge_group_df)
    print("\nLeaf -> coarse mapping:")
    display(mapping_df)

    return {
        "cm_df": cm_df,
        "pair_df": pair_df,
        "merge_groups": merge_groups,
        "merge_group_df": merge_group_df,
        "leaf_to_coarse": leaf_to_coarse,
        "mapping_df": mapping_df,
    }


# Run encoder selection and merge-group discovery

def run_encoder_selection():
    train_df, val_df, leaf_classes, split_counts = make_global_split()

    y_train_leaf = train_df["cls_label"].astype(str).values
    y_val_leaf = val_df["cls_label"].astype(str).values

    comparison_rows = []

    best_macro_f1 = -1.0
    best_dino_ckpt_path = None
    best_dino_info = None
    best_probe_pred = None
    best_probe_conf = None
    best_probe_epoch = None

    for dino_ckpt_path in DINO_CHECKPOINTS:
        if dino_ckpt_path is None:
            checkpoint_name = "original_siglip2_epoch0"
        else:
            checkpoint_name = dino_ckpt_path.name
    
        print("\n" + "=" * 100)
        print("Evaluating:", checkpoint_name)
        print("=" * 100)
    
        if dino_ckpt_path is not None and not dino_ckpt_path.exists():
            raise FileNotFoundError(f"Missing DINO checkpoint: {dino_ckpt_path}")

        set_seed(RANDOM_STATE)

        processor, encoder, dino_info = load_adapted_siglip_encoder(dino_ckpt_path)

        X_train = extract_siglip_embeddings_no_save(train_df, processor, encoder, "probe train")
        X_val = extract_siglip_embeddings_no_save(val_df, processor, encoder, "probe val")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        probe = train_probe_classifier(
            X_train_s=X_train_s,
            y_train=y_train_leaf,
            X_val_s=X_val_s,
            y_val=y_val_leaf,
        )

        candidate_macro_f1 = float(probe["best_macro_f1"])

        comparison_rows.append({
            "checkpoint": None if dino_ckpt_path is None else str(dino_ckpt_path),
            "checkpoint_name": checkpoint_name,
            "dino_epoch": dino_info["epoch"],
            "ssl_train_loss": dino_info["ssl_train_loss"],
            "probe_best_epoch": probe["best_epoch"],
            "probe_val_macro_f1": candidate_macro_f1,
        })

        print(
            f"Finished {checkpoint_name} | "
            f"DINO epoch={dino_info['epoch']} | "
            f"SSL loss={dino_info['ssl_train_loss']} | "
            f"Probe best epoch={probe['best_epoch']} | "
            f"Probe val macro-F1={candidate_macro_f1:.4f}"
        )

        if candidate_macro_f1 > best_macro_f1:
            best_macro_f1 = candidate_macro_f1
            best_dino_ckpt_path = None if dino_ckpt_path is None else Path(dino_ckpt_path)
            best_dino_info = dict(dino_info)
            best_probe_pred = probe["pred"].copy()
            best_probe_conf = probe["conf"].copy()
            best_probe_epoch = int(probe["best_epoch"])

        probe["model"].cpu()
        encoder.cpu()

        del probe
        del encoder
        del processor
        del scaler
        del X_train
        del X_val
        del X_train_s
        del X_val_s

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(comparison_rows).sort_values("probe_val_macro_f1", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(ENCODER_SELECTION_DIR / "dino_checkpoint_comparison.csv", index=False)

    print("\n========== DINO checkpoint comparison ==========")
    display(comparison_df)

    print("\n========== Selected checkpoint ==========")
    print("Checkpoint:", best_dino_ckpt_path)
    print("DINO epoch:", best_dino_info["epoch"])
    print("SSL train loss:", best_dino_info["ssl_train_loss"])
    print("Winning probe epoch:", best_probe_epoch)
    print("Winning probe val macro-F1:", best_macro_f1)

    winning_pred_df = pd.DataFrame({
        "image_path": val_df["image_path"].astype(str).values,
        "true_label": y_val_leaf,
        "pred_label": best_probe_pred,
        "confidence": best_probe_conf,
        "correct": y_val_leaf == best_probe_pred,
    })

    winning_pred_df.to_csv(ENCODER_SELECTION_DIR / "winning_probe_predictions.csv", index=False)

    merge_info = discover_merge_groups(
        y_true=y_val_leaf,
        y_pred=best_probe_pred,
        leaf_classes=leaf_classes,
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "leaf_classes": leaf_classes,
        "split_counts": split_counts,
        "best_dino_ckpt_path": best_dino_ckpt_path,
        "best_dino_info": best_dino_info,
        "comparison_df": comparison_df,
        "winning_probe_macro_f1": best_macro_f1,
        "winning_probe_epoch": best_probe_epoch,
        "merge_groups": merge_info["merge_groups"],
        "leaf_to_coarse": merge_info["leaf_to_coarse"],
        "merge_info": merge_info,
    }



