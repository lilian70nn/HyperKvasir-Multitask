import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .config import *
from .data import (
    build_segmentation_index,
    make_train_val_split,
    PolypSegmentationDataset,
    set_seed
)
from .model import (
    FrozenDinoSigLIP2SegmentationModel,
    segmentation_loss,
)

@torch.no_grad()
def compute_batch_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    tp = (predictions * targets).sum(dim=1)
    fp = (predictions * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - predictions) * targets).sum(dim=1)
    tn = ((1.0 - predictions) * (1.0 - targets)).sum(dim=1)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "pixel_accuracy": pixel_accuracy.cpu().numpy(),
    }


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    losses = []
    bce_losses = []
    dice_losses = []

    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []
    pixel_accuracy_values = []

    prediction_rows = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        targets = batch["mask"].to(DEVICE, non_blocking=True)

        logits = model(pixel_values)

        loss, bce, dice_loss = segmentation_loss(logits, targets)
        metrics = compute_batch_metrics(logits, targets, threshold=MASK_THRESHOLD)

        losses.append(float(loss.item()))
        bce_losses.append(float(bce.item()))
        dice_losses.append(float(dice_loss.item()))

        dice_values.extend(metrics["dice"].tolist())
        iou_values.extend(metrics["iou"].tolist())
        precision_values.extend(metrics["precision"].tolist())
        recall_values.extend(metrics["recall"].tolist())
        pixel_accuracy_values.extend(metrics["pixel_accuracy"].tolist())

        for i in range(len(batch["image_id"])):
            prediction_rows.append({
                "image_id": batch["image_id"][i],
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "dice": float(metrics["dice"][i]),
                "iou": float(metrics["iou"][i]),
                "precision": float(metrics["precision"][i]),
                "recall": float(metrics["recall"][i]),
                "pixel_accuracy": float(metrics["pixel_accuracy"][i]),
            })

    results = {
        "loss": float(np.mean(losses)),
        "bce_loss": float(np.mean(bce_losses)),
        "dice_loss": float(np.mean(dice_losses)),
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "pixel_accuracy": float(np.mean(pixel_accuracy_values)),
    }

    prediction_df = pd.DataFrame(prediction_rows)

    return results, prediction_df


@torch.no_grad()
def save_validation_predictions(model, dataset, epoch, max_images=20):
    out_dir = OUTPUT_DIR / "validation_predictions" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    count = min(max_images, len(dataset))

    for idx in range(count):
        sample = dataset[idx]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(DEVICE)
        logits = model(pixel_values)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prediction = (probability >= MASK_THRESHOLD).astype(np.uint8)

        Image.fromarray(prediction * 255).save(
            out_dir / f"{sample['image_id']}_pred.png"
        )



def train_segmentation_model(model, train_loader, val_loader, val_dataset):
    encoder_trainable = sum(
        p.numel()
        for p in model.encoder.parameters()
        if p.requires_grad
    )

    decoder_trainable = sum(
        p.numel()
        for p in model.decoder.parameters()
        if p.requires_grad
    )

    print("\n========== Trainable parameters ==========")
    print("Encoder trainable:", encoder_trainable)
    print("Decoder trainable:", f"{decoder_trainable:,}")

    if encoder_trainable != 0:
        raise RuntimeError("Encoder is supposed to be completely frozen.")

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_dice = -1.0
    best_epoch = -1
    patience = 0
    history = []

    best_checkpoint_path = OUTPUT_DIR / "best_segmentation_head.pt"

    print("\n========== Start segmentation training ==========")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        train_losses = []
        train_bce_losses = []
        train_dice_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{NUM_EPOCHS}",
        )

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            targets = batch["mask"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(pixel_values)
            loss, bce, dice = segmentation_loss(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError("Segmentation loss became NaN/Inf.")

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_bce_losses.append(float(bce.item()))
            train_dice_losses.append(float(dice.item()))

            pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}",
                bce=f"{np.mean(train_bce_losses):.4f}",
                dice=f"{np.mean(train_dice_losses):.4f}",
            )

        val_metrics, val_prediction_df = evaluate_model(model, val_loader)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_bce_loss": float(np.mean(train_bce_losses)),
            "train_dice_loss": float(np.mean(train_dice_losses)),
            "val_loss": val_metrics["loss"],
            "val_bce_loss": val_metrics["bce_loss"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "training_history.csv",
            index=False,
        )

        val_prediction_df.to_csv(
            OUTPUT_DIR / f"val_predictions_epoch_{epoch:03d}.csv",
            index=False,
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"Dice={row['val_dice']:.4f} | "
            f"IoU={row['val_iou']:.4f} | "
            f"Precision={row['val_precision']:.4f} | "
            f"Recall={row['val_recall']:.4f} | "
            f"PixelAcc={row['val_pixel_accuracy']:.4f}"
        )

        save_validation_predictions(
            model=model,
            dataset=val_dataset,
            epoch=epoch,
            max_images=20,
        )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_epoch = epoch
            patience = 0

            torch.save(
                {
                    "encoder": {
                        "model_name": ENCODER_INFO["model_name"],
                        "checkpoint_path": ENCODER_INFO["checkpoint_path"],
                        "checkpoint_state_key": ENCODER_INFO["checkpoint_state_key"],
                        "dino_epoch": ENCODER_INFO["dino_epoch"],
                        "ssl_train_loss": ENCODER_INFO["ssl_train_loss"],
                        "frozen": True,
                    },
                
                    "image_size": IMAGE_SIZE,
                    "patch_size": PATCH_SIZE,
                    "decoder_channels": DECODER_CHANNELS,
                    "dropout": DROPOUT,
                    "mask_threshold": MASK_THRESHOLD,
                    "best_epoch": best_epoch,
                    "best_val_dice": best_dice,
                
                    "decoder_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.decoder.state_dict().items()
                    },
                
                    "history": history.copy(),
                },
                best_checkpoint_path,
            )

            print("Updated best segmentation head:", best_checkpoint_path)

        else:
            patience += 1

        if patience >= PATIENCE:
            print(
                f"\nEarly stopping | "
                f"best_epoch={best_epoch} | "
                f"best_val_dice={best_dice:.4f}"
            )
            break

    print("\n========== Segmentation training complete ==========")
    print("Best epoch:", best_epoch)
    print("Best validation Dice:", best_dice)
    print("Best segmentation head:", best_checkpoint_path)

    return {
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_checkpoint_path": best_checkpoint_path,
    }


def run_segmentation_pipeline():
    set_seed(RANDOM_STATE)

    all_df = build_segmentation_index()
    train_df, val_df = make_train_val_split(all_df)

    print("\n========== Load SigLIP2 processor ==========")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    train_dataset = PolypSegmentationDataset(
        train_df,
        processor=processor,
        training=True,
    )

    val_dataset = PolypSegmentationDataset(
        val_df,
        processor=processor,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n========== Build frozen DINO-SigLIP2 segmentation model ==========")

    model = FrozenDinoSigLIP2SegmentationModel(
        model_name=MODEL_NAME,
        dino_checkpoint_path=DINO_CKPT_PATH,
        state_key=DINO_STATE_KEY,
        decoder_channels=DECODER_CHANNELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    results = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
    )

    return {
        "model": model,
        "processor": processor,
        "train_df": train_df,
        "val_df": val_df,
        **results,
    }


if __name__ == "__main__":
    segmentation_results = run_segmentation_pipeline()