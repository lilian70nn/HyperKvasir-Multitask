import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from .config import MODEL_NAME, UNFREEZE_LAST_VISION_LAYERS


# DINO projection head
class DinoProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Load SigLIP2 student / teacher
def load_siglip2_student_teacher():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    student = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    teacher = copy.deepcopy(student).to(device)

    for p in student.parameters():
        p.requires_grad = False

    vision_layers = student.vision_model.encoder.layers
    n_layers = len(vision_layers)

    if UNFREEZE_LAST_VISION_LAYERS < 0:
        raise ValueError("UNFREEZE_LAST_VISION_LAYERS cannot be negative.")

    if UNFREEZE_LAST_VISION_LAYERS > n_layers:
        raise ValueError(
            f"UNFREEZE_LAST_VISION_LAYERS={UNFREEZE_LAST_VISION_LAYERS} "
            f"> total vision layers={n_layers}"
        )

    if UNFREEZE_LAST_VISION_LAYERS > 0:
        for layer in vision_layers[-UNFREEZE_LAST_VISION_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True

    if hasattr(student.vision_model, "post_layernorm"):
        for p in student.vision_model.post_layernorm.parameters():
            p.requires_grad = True

    for p in teacher.parameters():
        p.requires_grad = False

    student.train()
    teacher.eval()

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())

    print(f"Device: {device}")
    print(f"Student trainable params: {trainable:,}")
    print(f"Student total params:     {total:,}")
    print(f"Trainable ratio:          {trainable / total:.6f}")

    return processor, student, teacher, device


# Encoding helpers

def encode_images_with_grad(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")

    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    outputs = model.get_image_features(**inputs)
    emb = outputs.pooler_output

    if emb is None:
        raise RuntimeError("SigLIP2 did not return pooler_output.")

    emb = F.normalize(emb.float(), dim=-1)
    return emb


@torch.no_grad()
def encode_images_no_grad(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")

    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    outputs = model.get_image_features(**inputs)
    emb = outputs.pooler_output

    if emb is None:
        raise RuntimeError("SigLIP2 did not return pooler_output.")

    emb = F.normalize(emb.float(), dim=-1)
    return emb


# EMA updates

@torch.no_grad()
def ema_update_teacher(student, teacher, momentum):
    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())

    if student_params.keys() != teacher_params.keys():
        raise RuntimeError("Student and teacher parameter structures do not match.")

    for name in teacher_params:
        teacher_params[name].data.mul_(momentum).add_(
            student_params[name].data,
            alpha=1.0 - momentum,
        )

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())

    for name in teacher_buffers:
        if name in student_buffers:
            teacher_buffers[name].data.copy_(student_buffers[name].data)


@torch.no_grad()
def ema_update_head(student_head, teacher_head, momentum):
    for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


# DINO-style loss

def dino_loss_one_direction(
    student_logits,
    teacher_logits,
    center,
    student_temp,
    teacher_temp,
):
    student_log_probs = F.log_softmax(
        student_logits / student_temp,
        dim=-1,
    )

    teacher_probs = F.softmax(
        (teacher_logits - center) / teacher_temp,
        dim=-1,
    ).detach()

    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss



# 10. Center update

@torch.no_grad()
def update_center(center, teacher_logits_list, momentum):
    batch_center = torch.cat(
        teacher_logits_list,
        dim=0,
    ).mean(dim=0, keepdim=True)

    center.mul_(momentum).add_(
        batch_center,
        alpha=1.0 - momentum,
    )

    return center