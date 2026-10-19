"""Stream sources. Mine: arrival schedules (fixed-rate, bursty), replay subset, webcam preprocessing.

The replay stream plays CIFAR-10 *test* images (labels known) at scheduled arrival times; a frame's
capture timestamp is its scheduled arrival, so any queueing behind a slow stage counts as latency.
CIFAR replay frames are independent images: there is no temporal correlation between frames.
"""
import threading
import time

import numpy as np

from ptq.data_cache import load, normalize

from .pipeline import Frame

REPLAY_SEED = 0


def replay_subset(n: int, seed: int = REPLAY_SEED) -> np.ndarray:
    """Fixed seeded subset of CIFAR-10 test indices, in stream order."""
    return np.random.default_rng(seed).permutation(10000)[:n]


def schedule(n: int, mode: str, fps: float = 30.0, seed: int = REPLAY_SEED) -> np.ndarray:
    """Arrival offsets (s). fixed: every 1/fps. bursty: bursts of ~Geom(mean 8) frames at 120 fps
    separated by gaps U(100, 400) ms (long-run average ~25-30 fps; gaps let the GPU go idle)."""
    if mode == "fixed":
        return np.arange(n) / fps
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while len(out) < n:
        for _ in range(rng.geometric(1 / 8)):
            out.append(t)
            t += 1 / 120
        t += rng.uniform(0.10, 0.40)
    return np.array(out[:n])


def replay_preprocess(raw_uint8_chw):
    return normalize(raw_uint8_chw[None])


def play_replay(pipe, n: int, mode: str, fps: float, seed: int = REPLAY_SEED):
    """Capture thread: submits frames at their scheduled times. Returns the thread."""
    x, y = load("test")
    idx = replay_subset(n, seed)
    offs = schedule(n, mode, fps, seed)
    frames = np.ascontiguousarray(x[np.sort(idx)])[np.argsort(np.argsort(idx))]  # preload into RAM:
    labels = y[idx]  # a camera hands over in-memory frames; disk reads must not count as latency.

    def run():
        t0 = time.perf_counter() + 0.05
        for fid, off in enumerate(offs):
            t_sched = t0 + off
            # macOS oversleeps by ~4-5 ms (timer slack): coarse sleep, then yield-spin to the deadline.
            while (d := t_sched - time.perf_counter()) > 0:
                time.sleep(d - 0.008 if d > 0.010 else 0)
            f = Frame(fid=fid, t_cap=t_sched, raw=frames[fid], label=int(labels[fid]))
            f.extra["t_sub"] = time.perf_counter()
            pipe.submit(f)

    th = threading.Thread(target=run, name="capture", daemon=True)
    th.start()
    return th


def webcam_preprocess(bgr):
    """BGR HxWx3 frame -> center square crop -> 224 bicubic -> CLIP-normalized (1,3,224,224)."""
    import cv2
    h, w = bgr.shape[:2]
    s = min(h, w)
    crop = bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return normalize(rgb.transpose(2, 0, 1)[None])
