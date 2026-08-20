from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/kelkalot/the-hyper-kvasir-dataset")
MODEL_NAME = "google/siglip2-base-patch16-224"
UNLABELED_IMAGE_DIR = DATA_ROOT / "unlabeled-images" / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
DINO_OUTPUT_DIR = Path("/kaggle/working/siglip2_dino_style_unlabeled_full")
DINO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DINO_MAX_UNLABELED_IMAGES = None
DINO_BATCH_SIZE = 32
DINO_NUM_WORKERS = 2
DINO_NUM_EPOCHS = 5
DINO_LR_VISION = 5e-6
DINO_LR_HEAD = 1e-4
DINO_WEIGHT_DECAY = 1e-4
UNFREEZE_LAST_VISION_LAYERS = 1
DINO_PROJ_HIDDEN_DIM = 1024
DINO_PROJ_OUT_DIM = 4096
STUDENT_TEMP = 0.10
TEACHER_TEMP = 0.04
EMA_MOMENTUM = 0.996
CENTER_MOMENTUM = 0.90
GRAD_CLIP_NORM = 1.0