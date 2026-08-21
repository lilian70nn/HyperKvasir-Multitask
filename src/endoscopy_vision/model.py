from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

from .classification import HierarchicalClassifier
from .segmentation import PolypSegmentor
from .config import (
    HF_REPO_ID,
    ENCODER_FILENAME,
    CLASSIFIER_FILENAME,
    SEGMENTATION_FILENAME,
    SIGLIP_MODEL_NAME,
    ENCODER_STATE_KEY,
    POLYP_LABEL,
)


class EndoscopyVisionModule:
    """Shared DINO-adapted SigLIP2 encoder with hierarchical classification and conditional polyp segmentation."""

    def __init__(
        self,
        encoder_checkpoint,
        classifier_checkpoint,
        segmentation_checkpoint,
        model_name=SIGLIP_MODEL_NAME,
        encoder_state_key=ENCODER_STATE_KEY,
        device=None,
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

        self.expected_encoder_info = self._check_task_encoder_consistency()

        self.model_name = self.expected_encoder_info["model_name"]
        self.encoder_state_key = self.expected_encoder_info["checkpoint_state_key"]

        self.processor, self.encoder = self._load_shared_encoder()

        self.classifier = HierarchicalClassifier(
            encoder=self.encoder,
            processor=self.processor,
            checkpoint_path=self.classifier_checkpoint,
            device=self.device,
        )

        self.segmentor = PolypSegmentor(
            encoder=self.encoder,
            processor=self.processor,
            checkpoint_path=self.segmentation_checkpoint,
            device=self.device,
        )

    def _check_task_encoder_consistency(self):
        classifier_ckpt = torch.load(self.classifier_checkpoint, map_location="cpu", weights_only=False)
        segmentation_ckpt = torch.load(self.segmentation_checkpoint, map_location="cpu", weights_only=False)

        if "encoder" not in classifier_ckpt:
            raise KeyError("Classifier checkpoint does not contain 'encoder' metadata.")

        if "encoder" not in segmentation_ckpt:
            raise KeyError("Segmentation checkpoint does not contain 'encoder' metadata.")

        cls_encoder = classifier_ckpt["encoder"]
        seg_encoder = segmentation_ckpt["encoder"]

        required_keys = ["model_name", "checkpoint_path", "checkpoint_state_key", "dino_epoch"]

        for key in required_keys:
            if key not in cls_encoder:
                raise KeyError(f"Classifier encoder metadata is missing '{key}'.")
            if key not in seg_encoder:
                raise KeyError(f"Segmentation encoder metadata is missing '{key}'.")

        mismatches = []

        for key in required_keys:
            if cls_encoder[key] != seg_encoder[key]:
                mismatches.append(
                    f"{key}: classification={cls_encoder[key]!r}, segmentation={seg_encoder[key]!r}"
                )

        if mismatches:
            raise RuntimeError(
                "Classification and segmentation checkpoints were trained with different encoders:\n"
                + "\n".join(mismatches)
            )

        print("\n========== Task encoder metadata verified ==========")
        print("Model:", cls_encoder["model_name"])
        print("Training checkpoint:", cls_encoder["checkpoint_path"])
        print("State key:", cls_encoder["checkpoint_state_key"])
        print("DINO epoch:", cls_encoder["dino_epoch"])

        if "ssl_train_loss" in cls_encoder:
            print("SSL train loss:", cls_encoder["ssl_train_loss"])

        del classifier_ckpt
        del segmentation_ckpt

        return dict(cls_encoder)

    def _load_shared_encoder(self):
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        encoder = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        checkpoint = torch.load(self.encoder_checkpoint, map_location="cpu", weights_only=False)

        if self.encoder_state_key not in checkpoint:
            raise KeyError(
                f"Encoder checkpoint does not contain '{self.encoder_state_key}'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        actual_model_name = checkpoint.get("model_name")
        expected_model_name = self.expected_encoder_info["model_name"]

        if actual_model_name is None:
            raise KeyError("Encoder checkpoint does not contain 'model_name' metadata.")

        if actual_model_name != expected_model_name:
            raise RuntimeError(
                f"Loaded encoder model mismatch: expected {expected_model_name!r}, got {actual_model_name!r}."
            )

        actual_epoch = checkpoint.get("epoch")
        expected_epoch = self.expected_encoder_info["dino_epoch"]

        if actual_epoch is None:
            raise KeyError("Encoder checkpoint does not contain 'epoch' metadata.")

        if int(actual_epoch) != int(expected_epoch):
            raise RuntimeError(
                f"Loaded encoder checkpoint mismatch: expected DINO epoch {expected_epoch}, got epoch {actual_epoch}."
            )

        expected_ssl_loss = self.expected_encoder_info.get("ssl_train_loss")
        actual_ssl_loss = checkpoint.get("ssl_train_loss")

        if expected_ssl_loss is not None and actual_ssl_loss is not None:
            if not np.isclose(float(expected_ssl_loss), float(actual_ssl_loss), rtol=1e-6, atol=1e-8):
                raise RuntimeError(
                    f"Loaded encoder SSL metadata mismatch: expected {expected_ssl_loss}, got {actual_ssl_loss}."
                )

        print("\n========== Shared encoder checkpoint verified ==========")
        print("Runtime checkpoint:", self.encoder_checkpoint)
        print("Model:", actual_model_name)
        print("State key:", self.encoder_state_key)
        print("DINO epoch:", actual_epoch)
        print("SSL train loss:", actual_ssl_loss)

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
    def _prepare_image(image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")

        raise TypeError("image must be a path, PIL.Image.Image, or numpy.ndarray")

    @torch.no_grad()
    def predict_classification(self, image):
        image = self._prepare_image(image)
        return self.classifier.predict(image)

    @torch.no_grad()
    def predict_segmentation(self, image, threshold=None):
        image = self._prepare_image(image)
        return self.segmentor.predict(image, threshold=threshold)

    @torch.no_grad()
    def predict(self, image, segmentation_threshold=None):
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

    def __call__(self, image, segmentation_threshold=None):
        return self.predict(image=image, segmentation_threshold=segmentation_threshold)



def load_endoscopy_model(
    encoder_checkpoint = None,
    classifier_checkpoint = None,
    segmentation_checkpoint = None,
    device = None,
):
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