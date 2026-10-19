"""Calibration-set selection from the training-split pool: seeded random vs greedy k-center
(farthest-point) coverage on L2-normalized FP32 CLIP image embeddings.

Each method produces one ordering of length max(CALIB_SIZES); the N-image calibration set is the
first N entries, so sets are nested across N."""
import json

import numpy as np
import torch

from .common import ART_DIR, CALIB_DIR, CALIB_SIZES, RANDOM_SEEDS, ImageEncoder, load_clip
from .data_cache import load, normalize

POOL_EMB = ART_DIR / "pool_emb_fp32.npy"


@torch.no_grad()
def embed_pool(batch_size: int = 128) -> np.ndarray:
    if POOL_EMB.exists():
        return np.load(POOL_EMB)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = ImageEncoder(load_clip()).to(device).eval()
    x, _ = load("pool")
    out = []
    for i in range(0, len(x), batch_size):
        out.append(enc(torch.from_numpy(normalize(x[i:i + batch_size])).to(device)).float().cpu().numpy())
    emb = np.concatenate(out)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(POOL_EMB, emb)
    return emb


def kcenter_order(emb: np.ndarray, k: int) -> np.ndarray:
    """Greedy k-center. Deterministic start: the pool point nearest the pool mean embedding."""
    start = int(np.argmin(np.linalg.norm(emb - emb.mean(0), axis=1)))
    order = [start]
    min_d = np.linalg.norm(emb - emb[start], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        order.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(emb - emb[nxt], axis=1))
    return np.array(order)


def random_order(n_pool: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_pool)[:k]


def coverage_radius(emb: np.ndarray, subset: np.ndarray) -> float:
    """Max distance from any pool point to its nearest selected point (the k-center objective)."""
    d = np.full(len(emb), np.inf)
    for s in subset:
        d = np.minimum(d, np.linalg.norm(emb - emb[s], axis=1))
    return float(d.max())


def build_selections() -> dict:
    """Returns {name: pool-position ordering}; also saves orderings + coverage stats."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIB_DIR / "orderings.json"
    if path.exists():
        return {k: np.array(v) for k, v in json.loads(path.read_text()).items()}
    emb = embed_pool()
    k = max(CALIB_SIZES)
    orders = {"kcenter": kcenter_order(emb, k)}
    for s in RANDOM_SEEDS:
        orders[f"random_s{s}"] = random_order(len(emb), k, s)
    _, labels = load("pool")
    stats = []
    for name, order in orders.items():
        for n in CALIB_SIZES:
            sub = order[:n]
            stats.append({"selection": name, "N": n, "coverage_radius": coverage_radius(emb, sub),
                          "n_classes": int(len(np.unique(labels[sub]))),
                          "class_counts": np.bincount(labels[sub], minlength=10).tolist()})
    path.write_text(json.dumps({k: v.tolist() for k, v in orders.items()}))
    (CALIB_DIR / "selection_stats.json").write_text(json.dumps(stats, indent=1))
    return orders
