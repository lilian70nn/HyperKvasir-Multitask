from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalClassifier:
    """
    Hierarchical classification branch.

    Pipeline:
        image
          -> shared DINO-adapted SigLIP2 encoder
          -> L2-normalized embedding
          -> saved StandardScaler
          -> coarse MLP
          -> direct leaf OR specialist Swin
          -> final leaf prediction

    The shared SigLIP2 encoder is supplied externally and is not loaded here.
    """

    def __init__(self, encoder: nn.Module, processor: Any, checkpoint_path: str | Path, device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder
        self.processor = processor
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Hierarchical classifier checkpoint not found: {self.checkpoint_path}")

        self.ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        self._load_scaler()
        self._load_coarse_head()
        self._load_hierarchy()
        self._load_specialists()

    def _load_scaler(self) -> None:
        scaler_info = self.ckpt["scaler"]
        self.scaler_mean = torch.as_tensor(scaler_info["mean"], dtype=torch.float32, device=self.device)
        self.scaler_scale = torch.as_tensor(scaler_info["scale"], dtype=torch.float32, device=self.device)

    def _load_coarse_head(self) -> None:
        info = self.ckpt["coarse_head"]
        self.coarse_classes = [str(x) for x in info["classes"]]

        self.coarse_head = MLPHead(
            input_dim=int(info["input_dim"]),
            hidden_dim=int(info["hidden_dim"]),
            num_classes=len(self.coarse_classes),
            dropout=float(info["dropout"]),
        )

        self.coarse_head.load_state_dict(info["state_dict"], strict=True)
        self.coarse_head.to(self.device).eval()

        for parameter in self.coarse_head.parameters():
            parameter.requires_grad = False

    def _load_hierarchy(self) -> None:
        hierarchy = self.ckpt["hierarchy"]
        self.leaf_classes = [str(x) for x in hierarchy["leaf_classes"]]
        self.leaf_to_coarse = {str(k): str(v) for k, v in hierarchy["leaf_to_coarse"].items()}
        self.merge_groups = [[str(x) for x in group] for group in hierarchy["merge_groups"]]

    def _load_specialists(self) -> None:
        self.specialist_models = nn.ModuleDict()
        self.specialist_processors: dict[str, Any] = {}
        self.specialist_classes: dict[str, list[str]] = {}
        self.merge_to_module_key: dict[str, str] = {}

        specialist_heads = self.ckpt.get("specialist_heads", {})

        for index, (merge_label, info) in enumerate(specialist_heads.items()):
            merge_label = str(merge_label)
            classes = [str(x) for x in info["label_classes"]]
            hf_model_name = str(info["hf_model_name"])

            id2label = {i: label for i, label in enumerate(classes)}
            label2id = {label: i for i, label in id2label.items()}

            image_processor = AutoImageProcessor.from_pretrained(hf_model_name)
            model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            model.load_state_dict(info["model_state_dict"], strict=True)
            model.to(self.device).eval()

            for parameter in model.parameters():
                parameter.requires_grad = False

            module_key = f"specialist_{index}"
            self.specialist_models[module_key] = model
            self.specialist_processors[merge_label] = image_processor
            self.specialist_classes[merge_label] = classes
            self.merge_to_module_key[merge_label] = module_key

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        outputs = self.encoder.get_image_features(**inputs)
        embedding = outputs.pooler_output

        if embedding is None:
            raise RuntimeError("SigLIP2 did not return pooler_output.")

        embedding = F.normalize(embedding.float(), dim=-1)
        embedding = (embedding - self.scaler_mean) / self.scaler_scale

        return embedding

    @torch.no_grad()
    def predict_coarse(self, image: Image.Image) -> dict[str, Any]:
        embedding = self.extract_embedding(image)
        logits = self.coarse_head(embedding)
        probabilities = F.softmax(logits, dim=1)[0]

        pred_index = int(probabilities.argmax().item())
        pred_label = self.coarse_classes[pred_index]
        confidence = float(probabilities[pred_index].item())

        return {"label": pred_label, "confidence": confidence}

    @torch.no_grad()
    def predict_specialist(self, image: Image.Image, merge_label: str) -> dict[str, Any]:
        if merge_label not in self.merge_to_module_key:
            raise KeyError(f"No specialist found for merge group: {merge_label}")

        module_key = self.merge_to_module_key[merge_label]
        model = self.specialist_models[module_key]
        image_processor = self.specialist_processors[merge_label]
        classes = self.specialist_classes[merge_label]

        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = model(pixel_values=pixel_values).logits
        probabilities = F.softmax(logits, dim=1)[0]

        pred_index = int(probabilities.argmax().item())
        pred_label = classes[pred_index]
        confidence = float(probabilities[pred_index].item())

        return {"label": pred_label, "confidence": confidence}

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("HierarchicalClassifier.predict expects a PIL.Image.Image.")

        image = image.convert("RGB")
        coarse_result = self.predict_coarse(image)

        coarse_label = coarse_result["label"]
        coarse_confidence = coarse_result["confidence"]

        if coarse_label not in self.merge_to_module_key:
            return {
                "label": coarse_label,
                "confidence": coarse_confidence,
                "route": "direct_leaf",
                "coarse_label": coarse_label,
                "coarse_confidence": coarse_confidence,
                "specialist_group": None,
                "specialist_confidence": None,
            }

        specialist_result = self.predict_specialist(image, coarse_label)
        specialist_label = specialist_result["label"]
        specialist_confidence = specialist_result["confidence"]
        final_confidence = coarse_confidence * specialist_confidence

        return {
            "label": specialist_label,
            "confidence": final_confidence,
            "route": "specialist",
            "coarse_label": coarse_label,
            "coarse_confidence": coarse_confidence,
            "specialist_group": coarse_label,
            "specialist_confidence": specialist_confidence,
        }

