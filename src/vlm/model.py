from pathlib import Path

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..endoscopy_vision.model import load_endoscopy_model


DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class EndoscopyAssistantModel:
    """
    General-purpose VLM with optional Endoscopy Vision Module support.

    Flow:
        text / image / both
            -> routing decision
            -> No: answer normally
            -> Yes: run Endoscopy Vision Module
                    -> classification + confidence + optional segmentation
                    -> use these results as evidence
                    -> generate final answer
                    -> highlight predicted region if segmentation exists
    """

    def __init__(self, model_name=DEFAULT_VLM_MODEL, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name

        print("Loading General Purpose VLM:", self.model_name)

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.vlm.to(self.device).eval()

        for parameter in self.vlm.parameters():
            parameter.requires_grad = False

        print("Loading Endoscopy Vision Module...")

        self.endoscopy_model = load_endoscopy_model(device=self.device)

        print("General Purpose VLM ready.")


    def _prepare_image(self, image):
        if image is None:
            return None

        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")

        raise TypeError("image must be a path, PIL.Image.Image, numpy.ndarray, or None")


    def _run_vlm(self, prompt, image=None, max_new_tokens=256):
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if image is None:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        else:
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")

        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        with torch.no_grad():
            generated = self.vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_length:]

        answer = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return answer.strip()


    def route(self, text=None, image=None):
        if image is None:
            return False

        user_text = text.strip() if text else "Describe and analyze the supplied image."

        prompt = f"""
    You are the routing component of a multimodal assistant.

    Decide whether the specialized Endoscopy Vision Module could provide useful domain-specific evidence for answering the current request.

    The Endoscopy Vision Module is designed for gastrointestinal endoscopy images. It performs finding classification and, when the predicted class is polyps, polyp segmentation.

    Use the specialist when:
    - the supplied image appears to be an endoscopy image, and
    - the user's request refers to, describes, identifies, interprets, captions, or analyzes the supplied image.

    The user does not need to explicitly ask for a diagnosis or classification. For requests such as "what's this?", "what do you see?", "describe this image", or "caption this image", the specialist classification can still provide useful evidence for the final answer.

    Do not use the specialist when:
    - the image is clearly unrelated to endoscopy, or
    - the user's request is unrelated to interpreting or describing the supplied image.

    User request:
    {user_text}

    Return exactly one word:
    YES
    or
    NO
    """

        decision = self._run_vlm(prompt, image=image, max_new_tokens=8).strip().upper()

        print(f"[router] decision = {decision!r}")

        if decision.startswith("YES"):
            return True

        if decision.startswith("NO"):
            return False

        return False

    def _evidence_text(self, result):
        classification = result.get("classification", {})
        segmentation = result.get("segmentation")

        lines = [
            f"Predicted endoscopy class: {result.get('label')}",
            f"Classification confidence: {result.get('confidence')}",
            f"Classification route: {classification.get('route')}",
        ]

        if classification.get("coarse_label") is not None:
            lines.append(f"Coarse prediction: {classification.get('coarse_label')}")
            lines.append(f"Coarse confidence: {classification.get('coarse_confidence')}")

        if classification.get("specialist_group") is not None:
            lines.append(f"Specialist group: {classification.get('specialist_group')}")
            lines.append(f"Specialist confidence: {classification.get('specialist_confidence')}")

        if segmentation is not None:
            mask = segmentation["mask"]
            probability = segmentation["probability"]

            lines.append("Polyp segmentation: available")
            lines.append(f"Segmentation threshold: {segmentation.get('threshold')}")
            lines.append(f"Predicted region fraction of image: {float(mask.mean()):.4f}")
            lines.append(f"Maximum segmentation probability: {float(probability.max()):.4f}")
        else:
            lines.append("Polyp segmentation: not available")

        return "\n".join(lines)


    def answer(self, text=None, image=None, evidence=None):
        if text:
            user_text = text.strip()
        elif image is not None:
            user_text = "Describe the supplied image and answer based on what can reasonably be inferred from it."
        else:
            raise ValueError("At least text or image must be provided.")

        if evidence is None:
            prompt = f"""
You are a helpful general-purpose multimodal assistant.

Answer the user's request using the available text and image. Respond naturally according to the user's question. If the image is medical, do not overstate certainty.

User request:
{user_text}
"""

        else:
            evidence_text = self._evidence_text(evidence)

            prompt = f"""
You are a helpful general-purpose multimodal assistant.

A specialized Endoscopy Vision Module has analyzed the supplied image. Use its result as additional evidence when answering the user's original request.

Use the evidence only when it is relevant to the question. Do not simply repeat every field. Answer naturally and directly. Treat the specialist prediction as supporting evidence rather than absolute ground truth. If segmentation is available, you may refer to the highlighted predicted region in the returned image.

User request:
{user_text}

Specialized endoscopy evidence:
{evidence_text}
"""

        return self._run_vlm(prompt, image=image, max_new_tokens=384)


    def highlight_mask(self, image, mask):
        image = self._prepare_image(image)

        image_array = np.asarray(image).astype(np.float32)
        mask = np.asarray(mask).astype(bool)

        if mask.shape != image_array.shape[:2]:
            resized_mask = Image.fromarray(mask.astype(np.uint8) * 255)
            resized_mask = resized_mask.resize(image.size, Image.Resampling.NEAREST)
            mask = np.asarray(resized_mask) > 127

        result = image_array.copy()

        highlight = np.zeros_like(result)
        highlight[..., 0] = 255
        highlight[..., 1] = 80

        alpha = 0.30
        result[mask] = result[mask] * (1.0 - alpha) + highlight[mask] * alpha

        padded = np.pad(mask, 1, constant_values=False)
        inside = (
            mask
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )

        boundary = mask & ~inside

        for _ in range(2):
            padded_boundary = np.pad(boundary, 1, constant_values=False)
            boundary = (
                padded_boundary[1:-1, 1:-1]
                | padded_boundary[:-2, 1:-1]
                | padded_boundary[2:, 1:-1]
                | padded_boundary[1:-1, :-2]
                | padded_boundary[1:-1, 2:]
            )

        result[boundary] = np.array([255, 255, 0], dtype=np.float32)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


    def predict(self, text=None, image=None):
        image = self._prepare_image(image)

        if text is None and image is None:
            raise ValueError("At least text or image must be provided.")

        use_endoscopy = self.route(text=text, image=image)

        if not use_endoscopy:
            answer = self.answer(text=text, image=image)

            return {
                "answer": answer,
                "annotated_image": None,
                "evidence": None,
                "used_endoscopy_vision": False,
            }

        evidence = self.endoscopy_model.predict(image)

        answer = self.answer(
            text=text,
            image=image,
            evidence=evidence,
        )

        annotated_image = None

        if evidence.get("segmentation") is not None:
            annotated_image = self.highlight_mask(
                image=image,
                mask=evidence["segmentation"]["mask"],
            )

        return {
            "answer": answer,
            "annotated_image": annotated_image,
            "evidence": evidence,
            "used_endoscopy_vision": True,
        }


    def __call__(self, text=None, image=None):
        return self.predict(text=text, image=image)