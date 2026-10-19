"""Full-test-set accuracy/embeddings for every variant, then isolated latency/memory benchmarks."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .build import int8_path
from .common import ART_DIR, CALIB_SIZES, ROOT, RANDOM_SEEDS, text_embeddings, logits_from_embeddings
from .data_cache import load, normalize
from .quantize import PREP_ONNX

EMB_DIR = ART_DIR / "emb"
BENCH_DIR = ART_DIR / "bench"


def variants():
    """name -> spec. Primary same-backend comparison: ORT CPU FP32 vs ORT CPU INT8."""
    v = {"fp32_ort_cpu": {"backend": "ort_cpu", "dtype": "fp32", "model": str(PREP_ONNX),
                          "selection": "-", "N": 0},
         "fp32_torch_mps": {"backend": "torch_mps", "dtype": "fp32", "selection": "-", "N": 0},
         "fp16_torch_mps": {"backend": "torch_mps", "dtype": "fp16", "selection": "-", "N": 0}}
    for sel in ["kcenter"] + [f"random_s{s}" for s in RANDOM_SEEDS]:
        for n in CALIB_SIZES:
            v[f"int8_{sel}_n{n}"] = {"backend": "ort_cpu", "dtype": "int8", "model": str(int8_path(sel, n)),
                                     "selection": sel, "N": n}
    return v


def embed_test(spec, batch_size: int = 64) -> np.ndarray:
    x, _ = load("test")
    if spec["backend"] == "ort_cpu":
        import onnxruntime as ort
        sess = ort.InferenceSession(spec["model"], providers=["CPUExecutionProvider"])
        run = lambda b: sess.run(None, {"pixel_values": b})[0]
    else:
        import torch
        from .common import ImageEncoder, load_clip
        dt = {"fp32": torch.float32, "fp16": torch.float16}[spec["dtype"]]
        enc = ImageEncoder(load_clip()).to("mps", dt).eval()
        run = lambda b: torch.no_grad()(enc)(torch.from_numpy(b).to("mps", dt)).float().cpu().numpy()
    return np.concatenate([run(normalize(x[i:i + batch_size])) for i in range(0, len(x), batch_size)])


def compute_embeddings():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in variants().items():
        path = EMB_DIR / f"{name}.npy"
        if path.exists():
            continue
        t = time.time()
        np.save(path, embed_test(spec).astype(np.float32))
        print(f"embedded test set with {name} in {time.time() - t:.0f}s", flush=True)


def run_benchmarks():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in variants().items():
        out = BENCH_DIR / f"{name}.json"
        if out.exists():
            continue
        cmd = [sys.executable, "-m", "ptq.bench", "--backend", spec["backend"], "--out", str(out)]
        cmd += ["--model", spec["model"]] if "model" in spec else ["--dtype", spec["dtype"]]
        subprocess.run(cmd, check=True, cwd=ROOT, stderr=subprocess.DEVNULL)
        r = json.loads(out.read_text())
        print(f"bench {name}: p50 {r['lat_p50_ms']:.2f} ms  p99 {r['lat_p99_ms']:.2f} ms  "
              f"peak RSS {r['peak_rss_mb']:.0f} MB", flush=True)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics():
    _, y = load("test")
    te = text_embeddings()
    ref = np.load(EMB_DIR / "fp32_ort_cpu.npy")
    ref_pred = logits_from_embeddings(ref, te).argmax(1)
    refn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    rows = []
    for name, spec in variants().items():
        e = np.load(EMB_DIR / f"{name}.npy")
        pred = logits_from_embeddings(e, te).argmax(1)
        cos = (e / np.linalg.norm(e, axis=1, keepdims=True) * refn).sum(1)
        b = json.loads((BENCH_DIR / f"{name}.json").read_text())
        model_mb = (Path(spec["model"]).stat().st_size / 2**20) if "model" in spec else None
        rows.append({"variant": name, "backend": spec["backend"], "precision": spec["dtype"],
                     "selection": spec["selection"], "N": spec["N"],
                     "top1_acc": float((pred == y).mean()), "agree_fp32": float((pred == ref_pred).mean()),
                     "cos_fp32_mean": float(cos.mean()), "cos_fp32_p01": float(np.percentile(cos, 1)),
                     "lat_mean_ms": b["lat_mean_ms"], "lat_p50_ms": b["lat_p50_ms"], "lat_p99_ms": b["lat_p99_ms"],
                     "peak_rss_mb": b["peak_rss_mb"], "rss_before_model_mb": b["rss_before_model_mb"],
                     "mps_driver_allocated_mb": b.get("mps_driver_allocated_mb"), "model_file_mb": model_mb})
    return rows
