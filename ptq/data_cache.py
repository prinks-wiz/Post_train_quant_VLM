"""Cache resized+center-cropped uint8 images so every stage (and the torch-free benchmark workers)
sees identical inputs. Normalization is applied on the fly by `normalize`."""
import json

import numpy as np

from .common import ART_DIR, CLIP_MEAN, CLIP_STD, POOL_SIZE, SEED, CIFAR10Parquet

CACHE_DIR = ART_DIR / "cache"
_MEAN = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(CLIP_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def normalize(x_uint8: np.ndarray) -> np.ndarray:
    return ((x_uint8.astype(np.float32) / 255.0) - _MEAN) / _STD


def _resize_crop(ds, indices, path):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(indices), 3, 224, 224))
    for j, i in enumerate(indices):
        out[j] = np.asarray(tf(ds.pil(int(i))), dtype=np.uint8).transpose(2, 0, 1)
    out.flush()


def pool_indices() -> np.ndarray:
    """Fixed 5,000-image calibration pool drawn from the CIFAR-10 *training* split only."""
    return np.sort(np.random.default_rng(SEED).choice(50000, POOL_SIZE, replace=False))


def build_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / "meta.json"
    if meta_path.exists():
        return
    train, test = CIFAR10Parquet(True), CIFAR10Parquet(False)
    pool = pool_indices()
    _resize_crop(train, pool, CACHE_DIR / "pool_uint8.npy")
    np.save(CACHE_DIR / "pool_labels.npy", np.array([train.targets[i] for i in pool]))
    np.save(CACHE_DIR / "pool_train_indices.npy", pool)
    _resize_crop(test, range(len(test)), CACHE_DIR / "test_uint8.npy")
    np.save(CACHE_DIR / "test_labels.npy", np.array(test.targets))
    meta_path.write_text(json.dumps({"pool_size": len(pool), "test_size": len(test)}))


def load(split: str):
    """split in {'pool', 'test'} -> (uint8 memmap (N,3,224,224), labels)."""
    return (np.load(CACHE_DIR / f"{split}_uint8.npy", mmap_mode="r"),
            np.load(CACHE_DIR / f"{split}_labels.npy"))
