# ============================================================
# Hierarchical classifier training and evaluation
#
# Purpose:
#   1. Load the encoder selection results, including the selected
#      SigLIP2 checkpoint, merge groups, and leaf-to-coarse mapping.
#   2. Re-extract train/validation embeddings using the selected encoder.
#   3. Fit a new scaler and train the coarse classifier on coarse labels.
#   4. Train one specialist classifier for each discovered merge group.
#   5. Route coarse merge-group predictions to the corresponding specialist.
#   6. Evaluate the complete hierarchical classifier at the leaf-class level.
#   7. Save coarse-classifier outputs, specialist results, hierarchical
#      predictions, and the exported hierarchical classifier checkpoint.
#
# Inputs from encoder selection:
#   global_train_df.csv
#   global_val_df.csv
#   selection_results.json
#
# Main output:
#   hierarchical_classifier.pt
#
# The probe classifier used during encoder selection is not reused here.
# The coarse classifier uses freshly extracted embeddings and a newly
# fitted StandardScaler.
# ============================================================

import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import gc
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display

from specialists import (
    SPECIALIST_BACKBONES,
    train_specialist,
    load_specialist,
    predict_specialist_paths,
)

from common import (
    OUTPUT_DIR,
    SIGLIP_MODEL_NAME,
    DEVICE,
    RANDOM_STATE,
    NUM_WORKERS,
    EMBED_BATCH_SIZE,
    MLP_HIDDEN_DIM,
    MLP_DROPOUT,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    MLP_PATIENCE,
    USE_CLASS_WEIGHT,
    set_seed,
    make_class_weights,
    compute_metrics,
    SigLIPImageDataset,
    siglip_collate_fn,
    load_adapted_siglip_encoder,
    MLPHead,
    predict_mlp,
    save_eval_outputs,
    make_merge_label
)

from encoder_selection import ENCODER_SELECTION_DIR

with open(ENCODER_SELECTION_DIR / "selection_results.json", "r") as f:
    selection_info = json.load(f)

merge_groups = selection_info["merge_groups"]
leaf_to_coarse = selection_info["leaf_to_coarse"]
leaf_classes = selection_info["leaf_classes"]

best_dino_ckpt_path = (
    None
    if selection_info["best_dino_ckpt_path"] is None
    else Path(selection_info["best_dino_ckpt_path"])
)

train_df = pd.read_csv(OUTPUT_DIR / "global_train_df.csv")
val_df = pd.read_csv(OUTPUT_DIR / "global_val_df.csv")

COARSE_DIR = OUTPUT_DIR / "coarse_classifier"
SPECIALIST_DIR = OUTPUT_DIR / "specialists"
HIERARCHICAL_DIR = OUTPUT_DIR / "hierarchical_classifier"

for d in [COARSE_DIR, SPECIALIST_DIR, HIERARCHICAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def copy_wrong_images(wrong_df, out_dir, max_per_pair=80):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(wrong_df) == 0:
        return

    for (true_label, pred_label), sub in tqdm(wrong_df.groupby(["true_label", "pred_label"]), desc=f"Copy wrong images -> {out_dir.name}"):
        sub = sub.sort_values("confidence", ascending=False).head(max_per_pair)
        pair_dir = out_dir / f"TRUE_{true_label}__PRED_{pred_label}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        for _, row in sub.iterrows():
            src = Path(row["image_path"])
            conf = float(row["confidence"])
            dst = pair_dir / f"conf_{conf:.3f}__{src.name}"

            if not dst.exists():
                shutil.copy2(src, dst)


def add_coarse_labels(df, leaf_to_coarse):
    out = df.copy()
    out["coarse_label"] = out["cls_label"].map(leaf_to_coarse)

    if out["coarse_label"].isna().any():
        missing = out.loc[out["coarse_label"].isna(), "cls_label"].unique().tolist()
        raise ValueError(f"Missing coarse mapping for: {missing}")

    return out


@torch.no_grad()
def extract_coarse_embeddings(df, processor, model, split_name):
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
    all_labels = []
    all_paths = []

    for images, labels, paths in tqdm(loader, desc=f"Coarse classifier embedding extraction: {split_name}"):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        emb = outputs.pooler_output

        if emb is None:
            raise RuntimeError("SigLIP2 did not return pooler_output")

        emb = F.normalize(emb.float(), dim=-1)

        all_embs.append(emb.cpu())
        all_labels.extend(labels)
        all_paths.extend(paths)

    X = torch.cat(all_embs, dim=0).numpy().astype(np.float32)

    return X, np.array(all_labels).astype(str), np.array(all_paths).astype(str)




def train_coarse_classifier(X_train_s, y_train, X_val_s, y_val):
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
    patience = 0
    history = []

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

        pred, conf, probs = predict_mlp(model, X_val_s, label_encoder)
        metrics = compute_metrics(y_val, pred)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        }

        history.append(row)

        print(
            f"Coarse classifier epoch {epoch:03d} | "
            f"loss={row['train_loss']:.4f} | "
            f"acc={row['accuracy']:.4f} | "
            f"bal_acc={row['balanced_accuracy']:.4f} | "
            f"macro_f1={row['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = float(metrics["f1_macro"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= MLP_PATIENCE:
            print(f"Coarse classifier early stopping | best_epoch={best_epoch} | best_macro_f1={best_macro_f1:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    pred, conf, probs = predict_mlp(model, X_val_s, label_encoder)

    pd.DataFrame(history).to_csv(COARSE_DIR / "training_history.csv", index=False)

    torch.save({
        "model_state_dict": model.state_dict(),
        "label_classes": label_encoder.classes_.tolist(),
        "input_dim": int(X_train_s.shape[1]),
        "hidden_dim": int(MLP_HIDDEN_DIM),
        "dropout": float(MLP_DROPOUT),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
    }, COARSE_DIR / "coarse_classifier.pt")

    return {
        "model": model,
        "label_encoder": label_encoder,
        "pred": pred,
        "conf": conf,
        "probs": probs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }


def run_hierarchical_pipeline():
    print("\n" + "=" * 100)
    print("HIERARCHICAL CLASSIFIER TRAINING")
    print("=" * 100)
    print("Selected encoder:", best_dino_ckpt_path)
    print("Merge groups:", merge_groups)

    set_seed(RANDOM_STATE)

    # --------------------------------------------------------
    # Fresh encoder load and fresh embedding extraction
    # --------------------------------------------------------

    siglip_processor, siglip_model, selected_dino_info = load_adapted_siglip_encoder(best_dino_ckpt_path)

    X_train, _, _ = extract_coarse_embeddings(
        train_df,
        siglip_processor,
        siglip_model,
        "train",
    )

    X_val, _, val_paths = extract_coarse_embeddings(
        val_df,
        siglip_processor,
        siglip_model,
        "val",
    )

    # --------------------------------------------------------
    # Fit a fresh scaler for the coarse classifier
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    siglip_model.cpu()
    del siglip_model
    del siglip_processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Build coarse labels from the fixed merge mapping
    # --------------------------------------------------------

    train_coarse_df = add_coarse_labels(train_df, leaf_to_coarse)
    val_coarse_df = add_coarse_labels(val_df, leaf_to_coarse)

    train_coarse_df.to_csv(COARSE_DIR / "train_df_with_coarse_label.csv", index=False)
    val_coarse_df.to_csv(COARSE_DIR / "val_df_with_coarse_label.csv", index=False)

    y_train_coarse = train_coarse_df["coarse_label"].astype(str).values
    y_val_coarse = val_coarse_df["coarse_label"].astype(str).values
    y_val_leaf = val_df["cls_label"].astype(str).values

    # --------------------------------------------------------
    # Train coarse classifier
    # --------------------------------------------------------

    coarse_model = train_coarse_classifier(
        X_train_s=X_train_s,
        y_train=y_train_coarse,
        X_val_s=X_val_s,
        y_val=y_val_coarse,
    )

    coarse_classes = coarse_model["label_encoder"].classes_.tolist()

    coarse_eval = save_eval_outputs(
        out_dir=COARSE_DIR,
        y_true=y_val_coarse,
        y_pred=coarse_model["pred"],
        y_conf=coarse_model["conf"],
        image_paths=val_paths,
        labels_for_cm=coarse_classes,
        prefix="coarse_",
    )

    print("\n========== Coarse classifier metrics ==========")
    display(coarse_eval["metrics_df"].T)

    # --------------------------------------------------------
    # Train specialists
    # --------------------------------------------------------

    specialist_infos = {}
    specialist_summaries = []

    for group_id, group_classes in enumerate(merge_groups, start=1):
        merge_label = make_merge_label(group_classes)

        candidate_infos = []

        for backbone_i, backbone_cfg in enumerate(SPECIALIST_BACKBONES):
            set_seed(RANDOM_STATE + group_id * 100 + backbone_i)

            info = train_specialist(
                group_id=group_id,
                merge_label=merge_label,
                group_classes=group_classes,
                train_df=train_df,
                val_df=val_df,
                backbone_tag=backbone_cfg["tag"],
                hf_model_name=backbone_cfg["hf_model_name"],
            )

            candidate_infos.append(info)

        best_info = sorted(
            candidate_infos,
            key=lambda x: (
                x["summary"]["val_f1_macro"],
                x["summary"]["val_balanced_accuracy"],
                x["summary"]["val_accuracy"],
            ),
            reverse=True,
        )[0]

        specialist_infos[merge_label] = best_info
        specialist_summaries.append(best_info["summary"])

    specialist_summary_df = pd.DataFrame(specialist_summaries)
    specialist_summary_df.to_csv(SPECIALIST_DIR / "selected_specialists.csv", index=False)

    print("\n========== Selected specialists ==========")
    display(specialist_summary_df)

    # --------------------------------------------------------
    # Hierarchical routing
    # --------------------------------------------------------

    coarse_pred = np.array(coarse_model["pred"]).astype(str)
    coarse_conf = np.array(coarse_model["conf"]).astype(float)

    hierarchical_pred = coarse_pred.copy()
    hierarchical_conf = coarse_conf.copy()

    hierarchical_route = np.array(["direct_leaf"] * len(val_df), dtype=object)
    hierarchical_specialist_conf = np.full(len(val_df), np.nan, dtype=float)
    hierarchical_specialist_group = np.array([""] * len(val_df), dtype=object)
    hierarchical_specialist_backbone = np.array([""] * len(val_df), dtype=object)

    for merge_label, info in specialist_infos.items():
        routed_idx = np.where(coarse_pred == merge_label)[0]

        print("\nRouting:", merge_label, "| N =", len(routed_idx))

        if len(routed_idx) == 0:
            continue

        routed_paths = val_paths[routed_idx]

        specialist_model, image_processor, label_encoder, ckpt = load_specialist(info["checkpoint_path"])

        specialist_pred, specialist_conf, specialist_probs = predict_specialist_paths(
            model=specialist_model,
            image_processor=image_processor,
            label_encoder=label_encoder,
            paths=routed_paths,
        )

        hierarchical_pred[routed_idx] = specialist_pred
        hierarchical_conf[routed_idx] = coarse_conf[routed_idx] * specialist_conf
        hierarchical_specialist_conf[routed_idx] = specialist_conf
        hierarchical_specialist_group[routed_idx] = merge_label
        hierarchical_specialist_backbone[routed_idx] = ckpt["backbone_tag"]
        hierarchical_route[routed_idx] = "specialist"

        specialist_model.cpu()
        del specialist_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Hierarchical leaf-level evaluation
    # --------------------------------------------------------

    hierarchical_eval = save_eval_outputs(
        out_dir=HIERARCHICAL_DIR,
        y_true=y_val_leaf,
        y_pred=hierarchical_pred,
        y_conf=hierarchical_conf,
        image_paths=val_paths,
        labels_for_cm=leaf_classes,
        prefix="hierarchical_",
    )

    hierarchical_pred_df = pd.DataFrame({
        "image_path": val_paths,
        "true_leaf": y_val_leaf,
        "true_coarse": y_val_coarse,
        "coarse_pred": coarse_pred,
        "coarse_confidence": coarse_conf,
        "route": hierarchical_route,
        "specialist_group": hierarchical_specialist_group,
        "specialist_backbone": hierarchical_specialist_backbone,
        "specialist_confidence": hierarchical_specialist_conf,
        "hierarchical_pred": hierarchical_pred,
        "hierarchical_confidence": hierarchical_conf,
        "hierarchical_correct": y_val_leaf == hierarchical_pred,
    })

    hierarchical_pred_df.to_csv(HIERARCHICAL_DIR / "hierarchical_predictions.csv", index=False)

    print("\n========== Hierarchical leaf-level metrics ==========")
    display(hierarchical_eval["metrics_df"].T)

    # --------------------------------------------------------
    # Export classifier.pt
    # --------------------------------------------------------

    specialist_heads = {}

    for merge_label, info in specialist_infos.items():
        specialist_ckpt = torch.load(info["checkpoint_path"], map_location="cpu", weights_only=False)

        specialist_heads[merge_label] = {
            "backbone_tag": specialist_ckpt["backbone_tag"],
            "hf_model_name": specialist_ckpt["hf_model_name"],
            "group_classes": list(specialist_ckpt["group_classes"]),
            "label_classes": list(specialist_ckpt["label_classes"]),
            "model_state_dict": {k: v.detach().cpu() for k, v in specialist_ckpt["model_state_dict"].items()},
        }

    classifier_ckpt = {
        "encoder": {
            "model_name": SIGLIP_MODEL_NAME,
            "checkpoint_path": str(best_dino_ckpt_path),
            "checkpoint_state_key": "student_state_dict",
            "dino_epoch": selected_dino_info["epoch"],
            "ssl_train_loss": selected_dino_info["ssl_train_loss"],
        },

        "scaler": {
            "mean": scaler.mean_.astype(np.float32),
            "scale": scaler.scale_.astype(np.float32),
        },

        "coarse_head": {
            "input_dim": int(X_train_s.shape[1]),
            "hidden_dim": int(MLP_HIDDEN_DIM),
            "dropout": float(MLP_DROPOUT),
            "classes": coarse_model["label_encoder"].classes_.tolist(),
            "state_dict": {k: v.detach().cpu() for k, v in coarse_model["model"].state_dict().items()},
        },

        "hierarchy": {
            "leaf_classes": list(leaf_classes),
            "leaf_to_coarse": dict(leaf_to_coarse),
            "merge_groups": [list(group) for group in merge_groups],
        },

        "specialist_heads": specialist_heads,
    }

    classifier_path = OUTPUT_DIR / "hierarchical_classifier.pt"
    torch.save(classifier_ckpt, classifier_path)

    print("\nSaved hierarchical classifier.pt:", classifier_path)

    return {
        "coarse_model": coarse_model,
        "coarse_eval": coarse_eval,
        "specialist_infos": specialist_infos,
        "specialist_summary_df": specialist_summary_df,
        "hierarchical_eval": hierarchical_eval,
        "hierarchical_pred_df": hierarchical_pred_df,
        "classifier_path": classifier_path,
    }


if __name__ == "__main__":
    hierarchical_results = run_hierarchical_pipeline()