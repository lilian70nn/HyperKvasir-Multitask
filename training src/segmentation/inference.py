from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .config import IMAGE_SIZE, MASK_THRESHOLD, DEVICE
from .model import FrozenDinoSigLIP2SegmentationModel

def load_trained_segmentation_model(
    segmentation_checkpoint,
    dino_checkpoint=None,
    device=None,
):
    device = torch.device(
        device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    segmentation_ckpt = torch.load(
        segmentation_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    encoder_info = segmentation_ckpt["encoder"]

    if dino_checkpoint is None:
        dino_checkpoint = encoder_info["checkpoint_path"]

    model = FrozenDinoSigLIP2SegmentationModel(
        model_name=encoder_info["model_name"],
        dino_checkpoint_path=dino_checkpoint,
        state_key=encoder_info["checkpoint_state_key"],
        decoder_channels=segmentation_ckpt["decoder_channels"],
        dropout=segmentation_ckpt["dropout"],
    )

    model.decoder.load_state_dict(
        segmentation_ckpt["decoder_state_dict"],
        strict=True,
    )

    model.to(device).eval()

    return model, segmentation_ckpt


@torch.no_grad()
def predict_segmentation(
    model,
    processor,
    image,
    threshold=MASK_THRESHOLD,
    device=DEVICE,
):
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")

    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    else:
        raise TypeError(
            "image must be a path or PIL.Image.Image"
        )

    original_height = image.height
    original_width = image.width

    resized = TF.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
    )

    processed = processor(
        images=resized,
        return_tensors="pt",
    )

    pixel_values = processed["pixel_values"].to(device)

    logits = model(pixel_values)

    probability = torch.sigmoid(logits)

    probability = F.interpolate(
        probability,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )

    probability = probability[0, 0].cpu().numpy()

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    return {
        "probability": probability,
        "mask": mask,
    }

@torch.no_grad()
def predict_and_show_segmentation(
    model,
    processor,
    image_path,
    threshold=MASK_THRESHOLD,
    device=DEVICE,
):
    result = predict_segmentation(
        model=model,
        processor=processor,
        image=image_path,
        threshold=threshold,
        device=device,
    )

    fig = show_segmentation_result(
        image_path=image_path,
        mask=result["mask"],
    )

    return {
        "probability": result["probability"],
        "mask": result["mask"],
        "figure": fig,
    }


def show_segmentation_result(image_path, mask):
    image = np.array(Image.open(image_path).convert("RGB"))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    ax.contour(
        mask,
        levels=[0.5],
        linewidths=2,
    )

    ax.axis("off")
    plt.show()

    return fig