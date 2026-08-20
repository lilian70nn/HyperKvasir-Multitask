import random
import torch
import numpy as np
import pandas as pd
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .config import *



def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_segmentation_index():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

    image_by_stem = {p.stem: p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    mask_by_stem = {p.stem: p for p in MASK_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}

    common_stems = sorted(set(image_by_stem) & set(mask_by_stem))

    if len(common_stems) == 0:
        raise ValueError("No matching image/mask pairs found.")

    rows = []
    for stem in common_stems:
        rows.append({
            "image_id": stem,
            "image_path": str(image_by_stem[stem]),
            "mask_path": str(mask_by_stem[stem]),
        })

    df = pd.DataFrame(rows)

    missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
    missing_images = sorted(set(mask_by_stem) - set(image_by_stem))

    print("\n========== Segmentation dataset ==========")
    print("Images:", len(image_by_stem))
    print("Masks:", len(mask_by_stem))
    print("Matched image/mask pairs:", len(df))
    print("Images without masks:", len(missing_masks))
    print("Masks without images:", len(missing_images))

    df.to_csv(OUTPUT_DIR / "all_segmentation_pairs.csv", index=False)

    return df


def make_train_val_split(df):
    train_df, val_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(OUTPUT_DIR / "train_df.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_df.csv", index=False)

    print("\n========== Train / validation split ==========")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))

    return train_df, val_df


def apply_train_transform(image, mask):
    image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BICUBIC)
    mask = TF.resize(mask, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.NEAREST)

    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if random.random() < 0.15:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    if random.random() < 0.30:
        angle = random.uniform(-8.0, 8.0)
        image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)

    if random.random() < 0.30:
        brightness = random.uniform(0.90, 1.10)
        contrast = random.uniform(0.90, 1.10)
        saturation = random.uniform(0.90, 1.10)

        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)

    return image, mask


def apply_eval_transform(image, mask):
    image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BICUBIC)
    mask = TF.resize(mask, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.NEAREST)
    return image, mask


class PolypSegmentationDataset(Dataset):
    def __init__(self, df, processor, training=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")

        with Image.open(row["mask_path"]) as m:
            mask = m.convert("L")

        if self.training and USE_AUGMENTATION:
            image, mask = apply_train_transform(image, mask)
        else:
            image, mask = apply_eval_transform(image, mask)

        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        mask_array = np.asarray(mask, dtype=np.float32)
        mask_array = (mask_array >= 127.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "mask_path": str(row["mask_path"]),
        }

