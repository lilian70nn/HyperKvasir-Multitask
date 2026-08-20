import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .config import *
from .data import (
    set_seed,
    collect_image_paths,
    build_dino_view_transform,
    UnlabeledDinoDataset,
    collate_dino_batch,
)
from .model import (
    DinoProjectionHead,
    load_siglip2_student_teacher,
    encode_images_with_grad,
    encode_images_no_grad,
    ema_update_teacher,
    ema_update_head,
    dino_loss_one_direction,
    update_center,
)

# Checkpoint saving

def save_epoch_checkpoint(
    epoch,
    avg_loss,
    student,
    teacher,
    student_head,
    teacher_head,
    center,
    logs,
):
    checkpoint_path = (
        DINO_OUTPUT_DIR
        / f"siglip2_dino_style_epoch{epoch}.pt"
    )

    checkpoint = {
        "model_name": MODEL_NAME,
        "method": "simplified_dino_style_siglip2_unlabeled_domain_adaptation",
        "epoch": int(epoch),
        "ssl_train_loss": float(avg_loss),

        "student_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in student.state_dict().items()
        },

        "teacher_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in teacher.state_dict().items()
        },

        "student_head_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in student_head.state_dict().items()
        },

        "teacher_head_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in teacher_head.state_dict().items()
        },

        "center": center.detach().cpu().clone(),

        "unfreeze_last_vision_layers": UNFREEZE_LAST_VISION_LAYERS,
        "dino_proj_hidden_dim": DINO_PROJ_HIDDEN_DIM,
        "dino_proj_out_dim": DINO_PROJ_OUT_DIM,
        "student_temp": STUDENT_TEMP,
        "teacher_temp": TEACHER_TEMP,
        "ema_momentum": EMA_MOMENTUM,
        "center_momentum": CENTER_MOMENTUM,
        "lr_vision": DINO_LR_VISION,
        "lr_head": DINO_LR_HEAD,
        "weight_decay": DINO_WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "batch_size": DINO_BATCH_SIZE,
        "num_epochs": DINO_NUM_EPOCHS,
        "random_state": RANDOM_STATE,

        "logs": logs.copy(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(f"Saved epoch {epoch} checkpoint: {checkpoint_path}")
    return checkpoint_path


# Main training

def run_dino_style_siglip2_unlabeled_training():
    print("========== Prepare unlabeled data ==========")

    image_paths = collect_image_paths(
        image_dir=UNLABELED_IMAGE_DIR,
        exts=IMAGE_EXTS,
        max_images=DINO_MAX_UNLABELED_IMAGES,
        random_state=RANDOM_STATE,
    )

    transform1 = build_dino_view_transform(strength="global")
    transform2 = build_dino_view_transform(strength="local")

    train_ds = UnlabeledDinoDataset(
        image_paths=image_paths,
        transform1=transform1,
        transform2=transform2,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=DINO_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=DINO_NUM_WORKERS,
        collate_fn=collate_dino_batch,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training DataLoader contains zero batches. "
            "Reduce DINO_BATCH_SIZE or provide more unlabeled images."
        )

    print(f"Training images: {len(train_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Epochs: {DINO_NUM_EPOCHS}")

    print("\n========== Load student / teacher ==========")

    processor, student, teacher, device = load_siglip2_student_teacher()

    if hasattr(student.config, "vision_config"):
        embedding_dim = int(student.config.vision_config.hidden_size)
    elif hasattr(student.vision_model.config, "hidden_size"):
        embedding_dim = int(student.vision_model.config.hidden_size)
    else:
        raise RuntimeError("Could not determine SigLIP2 embedding dimension.")

    print(f"Embedding dimension: {embedding_dim}")

    student_head = DinoProjectionHead(
        input_dim=embedding_dim,
        hidden_dim=DINO_PROJ_HIDDEN_DIM,
        output_dim=DINO_PROJ_OUT_DIM,
    ).to(device)

    teacher_head = copy.deepcopy(student_head).to(device)

    for p in teacher_head.parameters():
        p.requires_grad = False

    teacher_head.eval()

    center = torch.zeros(
        1,
        DINO_PROJ_OUT_DIM,
        device=device,
    )

    student_trainable_params = [
        p for p in student.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": student_trainable_params,
                "lr": DINO_LR_VISION,
            },
            {
                "params": student_head.parameters(),
                "lr": DINO_LR_HEAD,
            },
        ],
        weight_decay=DINO_WEIGHT_DECAY,
    )

    logs = []
    saved_checkpoints = []

    print("\n========== Start DINO-style training ==========")

    for epoch in range(1, DINO_NUM_EPOCHS + 1):
        student.train()
        student_head.train()
        teacher.eval()
        teacher_head.eval()

        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"DINO epoch {epoch}/{DINO_NUM_EPOCHS}",
        )

        for view1, view2, paths in pbar:
            optimizer.zero_grad(set_to_none=True)

            student_feat_1 = encode_images_with_grad(
                images=view1,
                processor=processor,
                model=student,
                device=device,
            )

            student_feat_2 = encode_images_with_grad(
                images=view2,
                processor=processor,
                model=student,
                device=device,
            )

            student_logits_1 = student_head(student_feat_1)
            student_logits_2 = student_head(student_feat_2)

            with torch.no_grad():
                teacher_feat_1 = encode_images_no_grad(
                    images=view1,
                    processor=processor,
                    model=teacher,
                    device=device,
                )

                teacher_feat_2 = encode_images_no_grad(
                    images=view2,
                    processor=processor,
                    model=teacher,
                    device=device,
                )

                teacher_logits_1 = teacher_head(teacher_feat_1)
                teacher_logits_2 = teacher_head(teacher_feat_2)

            loss_12 = dino_loss_one_direction(
                student_logits=student_logits_1,
                teacher_logits=teacher_logits_2,
                center=center,
                student_temp=STUDENT_TEMP,
                teacher_temp=TEACHER_TEMP,
            )

            loss_21 = dino_loss_one_direction(
                student_logits=student_logits_2,
                teacher_logits=teacher_logits_1,
                center=center,
                student_temp=STUDENT_TEMP,
                teacher_temp=TEACHER_TEMP,
            )

            loss = 0.5 * (loss_12 + loss_21)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"DINO-style loss became NaN/Inf during epoch {epoch}."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student_trainable_params
                + list(student_head.parameters()),
                max_norm=GRAD_CLIP_NORM,
            )

            optimizer.step()

            with torch.no_grad():
                ema_update_teacher(
                    student=student,
                    teacher=teacher,
                    momentum=EMA_MOMENTUM,
                )

                ema_update_head(
                    student_head=student_head,
                    teacher_head=teacher_head,
                    momentum=EMA_MOMENTUM,
                )

                center = update_center(
                    center=center,
                    teacher_logits_list=[
                        teacher_logits_1,
                        teacher_logits_2,
                    ],
                    momentum=CENTER_MOMENTUM,
                )

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)

            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{np.mean(epoch_losses):.4f}",
            )

        if len(epoch_losses) == 0:
            raise RuntimeError(
                f"No batches completed during epoch {epoch}."
            )

        avg_loss = float(np.mean(epoch_losses))

        print(
            f"\nEpoch {epoch}/{DINO_NUM_EPOCHS} complete | "
            f"DINO-style train loss = {avg_loss:.6f}"
        )

        log_row = {
            "epoch": epoch,
            "ssl_train_loss": avg_loss,
        }

        logs.append(log_row)

        log_df = pd.DataFrame(logs)

        log_df.to_csv(
            DINO_OUTPUT_DIR / "dino_style_training_log.csv",
            index=False,
        )

        print("\nTraining log:")
        print(log_df.to_string(index=False))

        checkpoint_path = save_epoch_checkpoint(
            epoch=epoch,
            avg_loss=avg_loss,
            student=student,
            teacher=teacher,
            student_head=student_head,
            teacher_head=teacher_head,
            center=center,
            logs=logs,
        )

        saved_checkpoints.append(checkpoint_path)

    print("\n========== Done DINO-style training ==========")
    print(f"Saved outputs to: {DINO_OUTPUT_DIR}")
    print("\nSaved checkpoints:")

    for checkpoint_path in saved_checkpoints:
        print(checkpoint_path)

    return {
        "processor": processor,
        "student": student,
        "teacher": teacher,
        "student_head": student_head,
        "teacher_head": teacher_head,
        "logs": pd.DataFrame(logs),
        "saved_checkpoints": saved_checkpoints,
    }