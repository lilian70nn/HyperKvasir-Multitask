from pathlib import Path
import torch

DATA_ROOT = Path("/kaggle/input/datasets/kelkalot/the-hyper-kvasir-dataset")
LABELED_ROOT = DATA_ROOT / "labeled-images"
LABEL_CSV = LABELED_ROOT / "image-labels.csv"

OUTPUT_DIR = Path("/kaggle/working/two_stage_leaf_mlp_coarse_swin_aug")
ENCODER_SELECTION_DIR = OUTPUT_DIR / "encoder_selection"
COARSE_DIR = OUTPUT_DIR / "coarse_classifier"
SPECIALIST_DIR = OUTPUT_DIR / "specialists"
HIERARCHICAL_DIR = OUTPUT_DIR / "hierarchical_classifier"

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
NUM_WORKERS = 2
TRAIN_RATIO = 0.80
MIN_CLASS_COUNT = 100
MERGE_THRESHOLD = 15

EMBED_BATCH_SIZE = 128

MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.25
MLP_BATCH_SIZE = 128
MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 15
USE_CLASS_WEIGHT = True

SPECIALIST_BACKBONES = [
    {"tag": "swin_tiny", "hf_model_name": "microsoft/swin-tiny-patch4-window7-224"},
]

SPECIALIST_BATCH_SIZE = 16
SPECIALIST_EPOCHS = 20
SPECIALIST_LR = 2e-5
SPECIALIST_WEIGHT_DECAY = 1e-4
SPECIALIST_PATIENCE = 6
USE_SPECIALIST_AUGMENTATION = True

DINO_CHECKPOINTS = [
    None,
    Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch1.pt"),
    Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch2.pt"),
    Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch3.pt"),
    Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch4.pt"),
    Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch5.pt"),
]

for directory in [OUTPUT_DIR, ENCODER_SELECTION_DIR, COARSE_DIR, SPECIALIST_DIR, HIERARCHICAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)