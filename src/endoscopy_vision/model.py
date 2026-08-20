from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

from .classification import TwoStageClassifier
from .segmentation import PolypSegmentor
from .config import HF_REPO_ID, ENCODER_FILENAME, CLASSIFIER_FILENAME, SEGMENTATION_FILENAME, SIGLIP_MODEL_NAME, ENCODER_STATE_KEY, POLYP_LABEL


class EndoscopyVisionModule:
    """Shared DINO-adapted SigLIP2 encoder with hierarchical classification and conditional polyp segmentation."""

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        classifier_checkpoint: str | Path,
        segmentation_checkpoint: str | Path,
        model_name: str = SIGLIP_MODEL_NAME,
        encoder_state_key: str = ENCODER_STATE_KEY,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.encoder_checkpoint = Path(encoder_checkpoint)
        self.classifier_checkpoint = Path(classifier_checkpoint)
        self.segmentation_checkpoint = Path(segmentation_checkpoint)

        if not self.encoder_checkpoint.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {self.encoder_checkpoint}")

        if not self.classifier_checkpoint.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {self.classifier_checkpoint}")

        if not self.segmentation_checkpoint.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {self.segmentation_checkpoint}")

        self.model_name = model_name
        self.encoder_state_key = encoder_state_key

        self.processor, self.encoder = self._load_shared_encoder()

        self.classifier = TwoStageClassifier(
            encoder=self.encoder,
            processor=self.processor,
            classifier_checkpoint=self.classifier_checkpoint,
            device=self.device,
        )

        self.segmentor = PolypSegmentor(
            encoder=self.encoder,
            processor=self.processor,
            segmentation_checkpoint=self.segmentation_checkpoint,
            device=self.device,
        )

    def _load_shared_encoder(self) -> tuple[Any, torch.nn.Module]:
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        encoder = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        checkpoint = torch.load(
            self.encoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if self.encoder_state_key not in checkpoint:
            raise KeyError(
                f"Encoder checkpoint does not contain '{self.encoder_state_key}'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        missing, unexpected = encoder.load_state_dict(
            checkpoint[self.encoder_state_key],
            strict=False,
        )

        if missing:
            print(f"[encoder] Missing keys: {len(missing)}")

        if unexpected:
            print(f"[encoder] Unexpected keys: {len(unexpected)}")

        for parameter in encoder.parameters():
            parameter.requires_grad = False

        encoder.to(self.device).eval()

        del checkpoint

        return processor, encoder

    @staticmethod
    def _prepare_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")

        raise TypeError("image must be a path, PIL.Image.Image, or numpy.ndarray")

    @torch.no_grad()
    def predict_classification(self, image: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
        image = self._prepare_image(image)
        return self.classifier.predict(image)

    @torch.no_grad()
    def predict_segmentation(
        self,
        image: str | Path | Image.Image | np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        image = self._prepare_image(image)
        return self.segmentor.predict(image, threshold=threshold)

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        segmentation_threshold: float | None = None,
    ) -> dict[str, Any]:
        image = self._prepare_image(image)

        classification = self.classifier.predict(image)

        result = {
            "label": classification["label"],
            "confidence": classification["confidence"],
            "classification": classification,
            "segmentation": None,
        }

        if classification["label"] == POLYP_LABEL:
            result["segmentation"] = self.segmentor.predict(
                image,
                threshold=segmentation_threshold,
            )

        return result

    def __call__(
        self,
        image: str | Path | Image.Image | np.ndarray,
        segmentation_threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict(
            image=image,
            segmentation_threshold=segmentation_threshold,
        )


def load_model(
    encoder_checkpoint: str | Path | None = None,
    classifier_checkpoint: str | Path | None = None,
    segmentation_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> EndoscopyVisionModule:
    if encoder_checkpoint is None:
        encoder_checkpoint = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=ENCODER_FILENAME,
        )

    if classifier_checkpoint is None:
        classifier_checkpoint = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CLASSIFIER_FILENAME,
        )

    if segmentation_checkpoint is None:
        segmentation_checkpoint = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SEGMENTATION_FILENAME,
        )

    return EndoscopyVisionModule(
        encoder_checkpoint=encoder_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        segmentation_checkpoint=segmentation_checkpoint,
        device=device,
    )