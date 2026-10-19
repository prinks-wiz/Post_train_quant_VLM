"""Lean latency + RAM worker: one model spec per process, no torch imported.

    python -m lean.bench --spec NAME --out JSON [--threads N] [--no-lean-mem]

RAM: rss_base (after importing numpy/cv2/onnxruntime), rss_loaded (after building sessions + head),
peak_rss (ru_maxrss, whole run). model_ram = rss_loaded - rss_base; peak_delta = peak - rss_base.
Latency: batch-1 on the first BENCH_N test images, preprocessing from raw 32x32 included; WARMUP untimed
runs; REPEATS passes, per-image median.
"""
import argparse
import json
import resource
import sys
import time

import numpy as np

BENCH_N, WARMUP, REPEATS = 300, 30, 3


def rss_now_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--no-lean-mem", action="store_true")
    ap.add_argument("--threshold", type=float)
    a = ap.parse_args()

    import onnxruntime  # noqa: F401  (import cost belongs to the baseline)
    from .runtime import LEAN_ART, Model, load_spec
    if load_spec(a.spec).get("prep") != "numpy":
        import cv2  # noqa: F401
    raw = np.load(LEAN_ART / "test_raw.npy", mmap_mode="r")
    frames = [np.array(raw[i]) for i in range(BENCH_N)]
    rss_base = rss_now_mb()
    spec = load_spec(a.spec)
    if a.threshold is not None:
        spec["threshold"] = a.threshold
    model = Model(spec, threads=a.threads, lean_mem=not a.no_lean_mem)
    rss_loaded = rss_now_mb()
    for i in range(WARMUP):
        model(frames[i % BENCH_N])
    lat = np.zeros((REPEATS, BENCH_N))
    escalated = np.zeros(BENCH_N, bool)
    for r in range(REPEATS):
        for i, f in enumerate(frames):
            t = time.perf_counter()
            _, esc = model(f)
            lat[r, i] = (time.perf_counter() - t) * 1e3
            escalated[i] = esc
    per = np.median(lat, 0)
    res = {"spec": a.spec, "threads": a.threads, "lean_mem": not a.no_lean_mem,
           "threshold": spec.get("threshold"), "lat_mean_ms": float(per.mean()),
           "lat_p50_ms": float(np.percentile(per, 50)), "lat_p99_ms": float(np.percentile(per, 99)),
           "frac_stage_b": float(escalated.mean()), "rss_base_mb": rss_base, "rss_loaded_mb": rss_loaded,
           "peak_rss_mb": peak_mb(), "model_ram_mb": rss_loaded - rss_base, "peak_delta_mb": peak_mb() - rss_base}
    with open(a.out, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main()
