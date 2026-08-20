from __future__ import annotations

import argparse

from .model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Endoscopy Vision Module inference")
    parser.add_argument("image", type=str, help="Path to an endoscopy image")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--threshold", type=float, default=None, help="Optional segmentation threshold")
    args = parser.parse_args()

    model = load_model(device=args.device)
    result = model.predict(args.image, segmentation_threshold=args.threshold)

    print(f"Predicted class: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")

    if result["segmentation"] is not None:
        print("Polyp segmentation: generated")


if __name__ == "__main__":
    main()