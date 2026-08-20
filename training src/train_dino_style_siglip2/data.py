import random
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .config import RANDOM_STATE


# Seed
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Augmentations
class GaussianBlurTransform:
    def __init__(self, radius_min=0.1, radius_max=1.2, p=0.2):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def build_dino_view_transform(strength="global"):
    if strength == "global":
        scale = (0.65, 1.0)
        color_jitter = transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015)
        blur_p = 0.15
    elif strength == "local":
        scale = (0.45, 0.90)
        color_jitter = transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)
        blur_p = 0.25
    else:
        raise ValueError(f"Unknown transform strength: {strength}")

    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=scale, ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=8),
        color_jitter,
        GaussianBlurTransform(radius_min=0.1, radius_max=1.2, p=blur_p),
    ])


# Collect unlabeled image paths
def collect_image_paths(image_dir, exts, max_images=None, random_state=42):
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = []
    for p in tqdm(image_dir.iterdir(), desc="Scanning unlabeled images"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)

    paths = sorted(paths)

    if len(paths) == 0:
        raise ValueError(f"No images found under: {image_dir}")

    if max_images is not None and len(paths) > max_images:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    print(f"Unlabeled images used for DINO-style training: {len(paths)}")
    return paths


# Dataset
class UnlabeledDinoDataset(Dataset):
    def __init__(self, image_paths, transform1, transform2):
        self.image_paths = [str(p) for p in image_paths]
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        view1 = self.transform1(image)
        view2 = self.transform2(image)
        return view1, view2, path


def collate_dino_batch(batch):
    view1 = [x[0] for x in batch]
    view2 = [x[1] for x in batch]
    paths = [x[2] for x in batch]
    return view1, view2, paths
