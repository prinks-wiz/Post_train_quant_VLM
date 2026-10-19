"""Shared setup: seeds, paths, CLIP model, CIFAR-10, preprocessing, text embeddings."""
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
MODEL_DIR = ART_DIR / "models"
CALIB_DIR = ART_DIR / "calib"
RESULTS_DIR = ROOT / "results"

MODEL_ID = "openai/clip-vit-base-patch32"
# Frozen zero-shot prompt template; do not change between calibration comparisons.
PROMPT_TEMPLATE = "a photo of a {}."
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

SEED = 0
POOL_SIZE = 5000
CALIB_SIZES = (32, 128, 512)
RANDOM_SEEDS = (0, 1, 2)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clip_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class CIFAR10Parquet(torch.utils.data.Dataset):
    """CIFAR-10 from the Hugging Face parquet mirror (uoft-cs/cifar10); same order/labels as the
    original release."""

    def __init__(self, train: bool, transform=None):
        import pandas as pd
        from huggingface_hub import hf_hub_download
        split = "train" if train else "test"
        path = hf_hub_download("uoft-cs/cifar10", f"plain_text/{split}-00000-of-00001.parquet",
                               repo_type="dataset", local_dir=DATA_DIR / "hf_cifar10")
        df = pd.read_parquet(path)
        self.images = [r["bytes"] for r in df["img"]]
        self.targets = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def pil(self, i):
        return Image.open(io.BytesIO(self.images[i])).convert("RGB")

    def __getitem__(self, i):
        img = self.pil(i)
        return (self.transform(img) if self.transform else img), self.targets[i]


def cifar10(train: bool):
    return CIFAR10Parquet(train, transform=clip_transform())


def stack_images(ds, indices) -> np.ndarray:
    return torch.stack([ds[int(i)][0] for i in indices]).numpy().astype(np.float32)


class ImageEncoder(nn.Module):
    """CLIP vision tower + projection -> unnormalized image embedding (B, 512)."""

    def __init__(self, clip):
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return self.visual_projection(pooled)


def load_clip():
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    return model.eval()


@torch.no_grad()
def text_embeddings(clip=None) -> np.ndarray:
    """Normalized FP32 class text embeddings (10, 512), cached to disk."""
    path = ART_DIR / "text_emb.npy"
    if path.exists():
        return np.load(path)
    from transformers import CLIPTokenizer
    clip = clip or load_clip()
    tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    inputs = tok([PROMPT_TEMPLATE.format(c) for c in CIFAR10_CLASSES], padding=True, return_tensors="pt")
    emb = clip.get_text_features(**inputs)
    emb = getattr(emb, "pooler_output", emb)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb.numpy())
    return emb.numpy()


def logits_from_embeddings(img_emb: np.ndarray, text_emb: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """FP32 cosine-similarity logits; scale matches CLIP's learned logit_scale (~100)."""
    img = img_emb / np.linalg.norm(img_emb, axis=-1, keepdims=True)
    return scale * img @ text_emb.T
