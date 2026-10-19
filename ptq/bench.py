"""Batch-1 latency + peak memory worker. Run as its own process per model so peak RSS is isolated:

    python -m ptq.bench --backend ort_cpu --model PATH --out JSON
    python -m ptq.bench --backend torch_mps --dtype fp16 --out JSON

Method: fixed first BENCH_N test images, WARMUP untimed runs, then REPEATS passes timing each image
once (wall clock, perf_counter; MPS synchronized). Per-image latency = median over passes."""
import argparse
import json
import resource
import time

import numpy as np

from .data_cache import load, normalize

BENCH_N, WARMUP, REPEATS = 300, 30, 3


def rss_mb():
    # macOS ru_maxrss is bytes (Linux: KiB).
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ort_cpu", "torch_mps", "torch_cpu"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    x_all, _ = load("test")
    xs = [normalize(x_all[i:i + 1]) for i in range(BENCH_N)]
    rss_before = rss_mb()
    extra = {}

    if a.backend == "ort_cpu":
        import onnxruntime as ort
        so = ort.SessionOptions()
        sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
        run = lambda x: sess.run(None, {"pixel_values": x})[0]
        extra["ort_version"] = ort.__version__
    else:
        import torch
        from .common import ImageEncoder, load_clip
        dev = "mps" if a.backend == "torch_mps" else "cpu"
        dt = {"fp32": torch.float32, "fp16": torch.float16}[a.dtype]
        enc = ImageEncoder(load_clip()).to(dev, dt).eval()
        sync = torch.mps.synchronize if dev == "mps" else (lambda: None)

        @torch.no_grad()
        def run(x):
            y = enc(torch.from_numpy(x).to(dev, dt)).float().cpu().numpy()
            sync()
            return y
        extra["torch_version"] = torch.__version__

    for i in range(WARMUP):
        run(xs[i % BENCH_N])
    lat = np.zeros((REPEATS, BENCH_N))
    for r in range(REPEATS):
        for i, x in enumerate(xs):
            t = time.perf_counter()
            run(x)
            lat[r, i] = (time.perf_counter() - t) * 1e3
    per_img = np.median(lat, axis=0)
    if a.backend == "torch_mps":
        import torch
        extra["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 2**20
    res = {"backend": a.backend, "dtype": a.dtype, "model": a.model,
           "lat_mean_ms": float(per_img.mean()), "lat_p50_ms": float(np.percentile(per_img, 50)),
           "lat_p99_ms": float(np.percentile(per_img, 99)), "per_image_ms": per_img.tolist(),
           "peak_rss_mb": rss_mb(), "rss_before_model_mb": rss_before,
           "bench_n": BENCH_N, "warmup": WARMUP, "repeats": REPEATS, **extra}
    with open(a.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
