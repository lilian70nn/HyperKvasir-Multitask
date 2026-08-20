import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import (
    MODEL_NAME,
    DINO_STATE_KEY,
    IMAGE_SIZE,
    PATCH_SIZE,
    DECODER_CHANNELS,
    DROPOUT,
    BCE_WEIGHT,
    DICE_WEIGHT,
    DINO_CKPT_PATH
)

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SegmentationDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_channels=256, dropout=0.10):
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

    def forward(self, feature_map, output_size=(224, 224)):
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


class FrozenDinoSigLIP2SegmentationModel(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        dino_checkpoint_path=DINO_CKPT_PATH,
        state_key=DINO_STATE_KEY,
        decoder_channels=DECODER_CHANNELS,
        dropout=DROPOUT,
    ):
        super().__init__()

        if not Path(dino_checkpoint_path).exists():
            raise FileNotFoundError(f"DINO checkpoint not found: {dino_checkpoint_path}")

        print("\n========== Load DINO-adapted SigLIP2 encoder ==========")

        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        checkpoint = torch.load(
            dino_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if state_key not in checkpoint:
            raise KeyError(
                f"DINO checkpoint does not contain '{state_key}'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        missing, unexpected = self.encoder.load_state_dict(checkpoint[state_key], strict=False)

        print("Loaded DINO checkpoint:", dino_checkpoint_path)
        print("DINO checkpoint epoch:", checkpoint.get("epoch"))
        print("DINO SSL train loss:", checkpoint.get("ssl_train_loss"))
        print("Missing encoder keys:", len(missing))
        print("Unexpected encoder keys:", len(unexpected))

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder.eval()

        if hasattr(self.encoder.config, "vision_config"):
            hidden_dim = int(self.encoder.config.vision_config.hidden_size)
        elif hasattr(self.encoder.vision_model.config, "hidden_size"):
            hidden_dim = int(self.encoder.vision_model.config.hidden_size)
        else:
            raise RuntimeError("Could not determine SigLIP2 vision hidden dimension.")

        self.hidden_dim = hidden_dim
        self.decoder = SegmentationDecoder(
            hidden_dim=hidden_dim,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        del checkpoint
        gc.collect()

    def train(self, mode=True):
        super().train(mode)

        # Encoder must stay frozen and in eval mode.
        self.encoder.eval()

        return self

    def tokens_to_feature_map(self, tokens):
        batch_size, token_count, hidden_dim = tokens.shape

        expected_grid = IMAGE_SIZE // PATCH_SIZE
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
                raise RuntimeError(
                    f"Cannot reshape {token_count} vision tokens into a square feature map."
                )

            patch_tokens = tokens

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_size,
            grid_size,
        )

        return feature_map

    def forward(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.encoder.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

            tokens = vision_outputs.last_hidden_state

        feature_map = self.tokens_to_feature_map(tokens)

        logits = self.decoder(
            feature_map,
            output_size=(IMAGE_SIZE, IMAGE_SIZE),
        )

        return logits


def dice_loss_from_logits(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return (1.0 - dice_score).mean()


def segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)

    total = BCE_WEIGHT * bce + DICE_WEIGHT * dice

    return total, bce, dice