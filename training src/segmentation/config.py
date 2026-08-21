from pathlib import Path
import torch


OUTPUT_DIR = Path("/kaggle/working/frozen_dino_siglip2_polyp_segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/kaggle/input/datasets/kelkalot/the-hyper-kvasir-dataset")
SEG_ROOT = DATA_ROOT / "segmented-images"
IMAGE_DIR = SEG_ROOT / "images"
MASK_DIR = SEG_ROOT / "masks"

CLASSIFIER_CKPT_PATH = Path(OUTPUT_DIR / "hierarchical_classifier.pt")
classifier_ckpt = torch.load(
    CLASSIFIER_CKPT_PATH,
    map_location="cpu",
    weights_only=False,
)
ENCODER_INFO = classifier_ckpt["encoder"]

MODEL_NAME = ENCODER_INFO["model_name"]
DINO_CKPT_PATH = Path(ENCODER_INFO["checkpoint_path"])
DINO_STATE_KEY = ENCODER_INFO["checkpoint_state_key"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

