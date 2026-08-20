from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationDecoder(nn.Module):
    def __init__(self, hidden_dim: int, decoder_channels: int = 256, dropout: float = 0.10):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )

        self.up1 = ConvNormAct(decoder_channels, 256)
        self.up2 = ConvNormAct(256, 128)
        self.up3 = ConvNormAct(128, 64)
        self.up4 = ConvNormAct(64, 32)

        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feature_map: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.projection(feature_map)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)

        x = self.dropout(x)
        logits = self.head(x)

        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return logits


class PolypSegmentor:
    """
    Segmentation branch using the shared DINO-adapted SigLIP2 encoder.

    Pipeline:
        image
          -> shared SigLIP2 vision encoder
          -> patch-token feature map
          -> trained segmentation decoder
          -> probability mask
          -> binary mask

    The shared encoder and processor are supplied externally.
    The segmentation checkpoint contains decoder weights only.
    """

    def __init__(
        self,
        encoder: nn.Module,
        processor: Any,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder
        self.processor = processor
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {self.checkpoint_path}")

        self.ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        self.image_size = int(self.ckpt.get("image_size", 224))
        self.patch_size = int(self.ckpt.get("patch_size", 16))
        self.mask_threshold = float(self.ckpt.get("mask_threshold", 0.5))

        self._load_decoder()

    def _get_hidden_dim(self) -> int:
        if hasattr(self.encoder.config, "vision_config"):
            return int(self.encoder.config.vision_config.hidden_size)

        if hasattr(self.encoder, "vision_model") and hasattr(self.encoder.vision_model.config, "hidden_size"):
            return int(self.encoder.vision_model.config.hidden_size)

        raise RuntimeError("Could not determine SigLIP2 vision hidden dimension.")

    def _load_decoder(self) -> None:
        hidden_dim = self._get_hidden_dim()
        decoder_channels = int(self.ckpt.get("decoder_channels", 256))
        dropout = float(self.ckpt.get("dropout", 0.10))

        self.decoder = SegmentationDecoder(
            hidden_dim=hidden_dim,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        self.decoder.load_state_dict(self.ckpt["decoder_state_dict"], strict=True)
        self.decoder.to(self.device).eval()

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

    def tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, hidden_dim = tokens.shape

        expected_grid = self.image_size // self.patch_size
        expected_tokens = expected_grid * expected_grid

        if token_count == expected_tokens:
            patch_tokens = tokens
            grid_size = expected_grid

        elif token_count == expected_tokens + 1:
            patch_tokens = tokens[:, 1:, :]
            grid_size = expected_grid

        else:
            grid_size = int(round(token_count ** 0.5))

            if grid_size * grid_size != token_count:
                raise RuntimeError(f"Cannot reshape {token_count} vision tokens into a square feature map.")

            patch_tokens = tokens

        return patch_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, grid_size, grid_size)

    @torch.no_grad()
    def extract_feature_map(self, image: Image.Image) -> torch.Tensor:
        resized = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
        )

        inputs = self.processor(images=resized, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        vision_outputs = self.encoder.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        tokens = vision_outputs.last_hidden_state

        return self.tokens_to_feature_map(tokens)

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, threshold: float | None = None) -> dict[str, Any]:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                image = img.convert("RGB")

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        else:
            raise TypeError("image must be a path or PIL.Image.Image")

        original_width, original_height = image.size
        threshold = self.mask_threshold if threshold is None else float(threshold)

        feature_map = self.extract_feature_map(image)

        logits = self.decoder(
            feature_map,
            output_size=(self.image_size, self.image_size),
        )

        probability = torch.sigmoid(logits)

        probability = F.interpolate(
            probability,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        probability = probability[0, 0].cpu().numpy()
        mask = (probability >= threshold).astype(np.uint8)

        return {
            "probability": probability,
            "mask": mask,
            "threshold": threshold,
        }

    @torch.no_grad()
    def predict_contour(self, image: str | Path | Image.Image, threshold: float | None = None) -> dict[str, Any]:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                pil_image = img.convert("RGB")

        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")

        else:
            raise TypeError("image must be a path or PIL.Image.Image")

        result = self.predict(pil_image, threshold=threshold)

        return {
            "image": pil_image,
            "probability": result["probability"],
            "mask": result["mask"],
            "threshold": result["threshold"],
        }

    def __call__(self, image: str | Path | Image.Image, threshold: float | None = None) -> dict[str, Any]:
        return self.predict(image, threshold=threshold)