# HyperKvasir Multitask

A multimodal endoscopy assistant that combines a general-purpose vision-language model with a specialized endoscopy vision module for hierarchical classification and conditional polyp segmentation.

The system accepts **text, an image, or both**. A general-purpose VLM first determines whether specialized endoscopy analysis is required. If required, the image is passed to the Endoscopy Vision Module, which performs endoscopic finding classification and conditionally performs polyp segmentation. The resulting structured evidence is then returned to the VLM to generate the final response.

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="100%">
</p>

## Endoscopy Vision Module

The specialist vision module uses a shared DINO-style adapted SigLIP2 encoder.

### Hierarchical Classification

The classification pipeline is:

```text
Image
  ↓
DINO-style adapted SigLIP2
  ↓
L2-normalized visual embedding
  ↓
StandardScaler
  ↓
Coarse MLP classifier
  ↓
Direct leaf prediction
or
Specialist Swin classifier
  ↓
Final class
```

The specialist classifiers are only activated for coarse classes that contain multiple visually similar leaf classes.

### Polyp Segmentation

When the final predicted class is `polyps`, the segmentation branch is activated:

```text
Image
  ↓
Shared SigLIP2 vision encoder
  ↓
Patch-token feature map
  ↓
Segmentation decoder
  ↓
Polyp probability map
  ↓
Binary segmentation mask
```

The predicted mask can additionally be overlaid on the original image to highlight the detected polyp region.

## Project Structure

```text
HyperKvasir-Multitask/
│
├── assets/
│   └── architecture.png                 # System architecture diagram
│
├── src/                                # Runtime inference modules
│   │
│   ├── endoscopy_vision/               # Specialist endoscopy vision module
│   │   ├── classification.py           # Hierarchical classifier: coarse MLP + specialist Swin models
│   │   ├── segmentation.py             # Polyp segmentation decoder and mask prediction
│   │   ├── model.py                    # Shared SigLIP2 encoder and classification/segmentation orchestration
│   │   └── config.py                   # Hugging Face checkpoints and inference configuration
│   │
│   └── vlm/
│       └── model.py                    # General-purpose VLM, routing, evidence integration, and final response generation
│
├── training src/                       # Model development and training pipelines
│   │
│   ├── classification/                 # Hierarchical endoscopy classification
│   │   ├── common.py                   # Shared classification utilities
│   │   ├── encoder_selection.py        # Encoder evaluation/selection experiments
│   │   ├── hierarchical_classifier.py  # Coarse hierarchical classifier training
│   │   ├── specialists.py              # Specialist classifiers for merged class groups
│   │   ├── siglip-cls-two-stage.ipynb  # Classification experiments and development notebook
│   │   └── results/                    # Classification experiment outputs
│   │
│   ├── segmentation/                   # Polyp segmentation training pipeline
│   │   ├── config.py                   # Segmentation training configuration
│   │   ├── data.py                     # Segmentation dataset and preprocessing
│   │   ├── model.py                    # Segmentation model/decoder definition
│   │   ├── train.py                    # Segmentation training loop
│   │   ├── inference.py                # Segmentation evaluation/inference utilities
│   │   ├── dino-siglip2-segmentation.ipynb
│   │   └── results/                    # Segmentation experiment outputs
│   │
│   └── train_dino_style_siglip2/       # DINO-style SigLIP2 adaptation
│       ├── config.py                    # Encoder adaptation configuration
│       ├── data.py                     # Training data pipeline
│       ├── model.py                    # Teacher/student model definitions
│       ├── train.py                    # DINO-style adaptation training
│       ├── dinov2-style-siglip.ipynb   # Encoder adaptation experiments
│       └── results/                    # Encoder training outputs
│
├── app.py                              # Gradio web interface
├── requirements.txt                    # Runtime dependencies
└── README.md                           # Project documentation
```

The repository separates **runtime inference** from **model training and experimentation**. `src/` contains only the components required by the deployed assistant, while `training src/` contains the pipelines used to develop the adapted encoder, hierarchical classifier, specialist classifiers, and segmentation model.

At runtime, `src/vlm/model.py` acts as the top-level orchestration layer. It determines whether specialist endoscopy analysis is required, invokes `src/endoscopy_vision/` when necessary, integrates the returned structured evidence, and produces the final multimodal response. `app.py` exposes this pipeline through the interactive Gradio interface.

## Model Checkpoints

The trained model checkpoints are hosted on Hugging Face and are downloaded automatically when the Endoscopy Vision Module is initialized.

Repository:

`Lian70/HyperKvasir-Multitask`

The specialist module uses:

- DINO-style adapted SigLIP2 encoder
- Hierarchical classifier
- Polyp segmentation decoder

The base vision encoder is:

`google/siglip2-base-patch16-224`

The general-purpose multimodal model is:

`Qwen/Qwen2.5-VL-3B-Instruct`

## Installation

Clone the repository:

```bash
git clone https://github.com/lilian70nn/HyperKvasir-Multitask.git
cd HyperKvasir-Multitask
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended because both the general-purpose VLM and the specialist endoscopy models are loaded during inference.

## Running the Application

Start the Gradio application:

```bash
python app.py
```

The application allows you to provide:

- text only
- an image only
- both text and an image

For local execution, open the URL printed by Gradio.

When running on a remote environment such as Google Colab, launch Gradio with sharing enabled and open the generated `gradio.live` URL.

## Python Usage

The Endoscopy Vision Module can also be used independently:

```python
from src.endoscopy_vision.model import load_model

model = load_model()

result = model.predict("example.jpg")

print(result["label"])
print(result["confidence"])

if result["segmentation"] is not None:
    mask = result["segmentation"]["mask"]
```

The complete multimodal assistant can be used through:

```python
from src.vlm.model import EndoscopyAssistantModel

model = EndoscopyAssistantModel()

result = model.predict(
    text="What do you see in this endoscopy image?",
    image="example.jpg",
)

print(result["answer"])

if result["annotated_image"] is not None:
    result["annotated_image"].show()
```

The assistant returns structured output containing the generated answer, optional specialist evidence, and an optional annotated image when polyp segmentation is available.

## Training

Training code is provided under `training src/` and is separated from the inference pipeline.

It contains the training workflows for:

- DINO-style SigLIP2 adaptation
- hierarchical endoscopy classification
- polyp segmentation

The runtime application only depends on the models under `src/`.

