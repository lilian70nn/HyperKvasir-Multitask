from pathlib import Path
import torch

DATA_ROOT = Path("/kaggle/input/datasets/kelkalot/the-hyper-kvasir-dataset")
SEG_ROOT = DATA_ROOT / "segmented-images"
IMAGE_DIR = SEG_ROOT / "images"
MASK_DIR = SEG_ROOT / "masks"

DINO_CKPT_PATH = Path("/kaggle/input/notebooks/lilyii70/dinov2-style-siglip/siglip2_dino_style_unlabeled_full/siglip2_dino_style_epoch3.pt")
DINO_STATE_KEY = "student_state_dict"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_NAME = "google/siglip2-base-patch16-224"

TRAIN_RATIO = 0.80

IMAGE_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 16
NUM_WORKERS = 2

RANDOM_STATE = 42
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

MASK_THRESHOLD = 0.5

DECODER_CHANNELS = 256
DROPOUT = 0.10

USE_AUGMENTATION = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("/kaggle/working/frozen_dino_siglip2_polyp_segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)