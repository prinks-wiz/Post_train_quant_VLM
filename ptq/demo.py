"""Interactive-ish demo of the confidence router (INT8 ORT CPU -> FP16 MPS on low margin).

    python -m ptq.demo path/to/image.jpg [more.jpg ...]
    python -m ptq.demo --cifar-test 0 1 2 3
    python -m ptq.demo --bench            # real end-to-end routed timing on the benchmark images

Prints the INT8 prediction, margin, escalation decision, final prediction and measured latency.
"""
import argparse
import json
import time

import numpy as np

from .common import CIFAR10_CLASSES, RESULTS_DIR, logits_from_embeddings, text_embeddings
from .data_cache import load, normalize


class Router:
    def __init__(self, threshold: float):
        import onnxruntime as ort
        import torch
        from .build import int8_path
        from .common import ImageEncoder, load_clip
        self.torch = torch
        self.int8 = ort.InferenceSession(str(int8_path("kcenter", 512)), providers=["CPUExecutionProvider"])
        self.fp16 = ImageEncoder(load_clip()).to("mps", torch.float16).eval()
        self.te = text_embeddings()
        self.threshold = threshold

    def __call__(self, x):
        t = time.perf_counter()
        lg = logits_from_embeddings(self.int8.run(None, {"pixel_values": x})[0], self.te)[0]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        top2 = np.sort(p)[-2:]
        m = float(top2[1] - top2[0])
        int8_pred = int(lg.argmax())
        esc = m < self.threshold
        final = int8_pred
        if esc:
            with self.torch.no_grad():
                e = self.fp16(self.torch.from_numpy(x).to("mps", self.torch.float16)).float().cpu().numpy()
            final = int(logits_from_embeddings(e, self.te)[0].argmax())
        return {"int8_pred": int8_pred, "margin": m, "escalated": esc, "final_pred": final,
                "latency_ms": (time.perf_counter() - t) * 1e3}


def default_threshold(frac: float = 0.10) -> float:
    """Threshold that escalates ~`frac` of the test set (from results/routing.csv)."""
    import pandas as pd
    df = pd.read_csv(RESULTS_DIR / "routing.csv")
    df = df[df.target == "fp16_torch_mps"]
    return float(df.iloc[(df.frac_escalated - frac).abs().argmin()].threshold)


def preprocess_file(path):
    from PIL import Image
    from .common import clip_transform
    return clip_transform()(Image.open(path).convert("RGB")).unsqueeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--cifar-test", type=int, nargs="*", default=[])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    thr = a.threshold if a.threshold is not None else default_threshold()
    router = Router(thr)
    x_test, y_test = load("test")
    for _ in range(20):  # warm both paths
        router.fp16(router.torch.from_numpy(normalize(x_test[:1])).to("mps", router.torch.float16))
        router(normalize(x_test[:1]))
    print(f"router: INT8 k-center N=512 (ORT CPU) -> FP16 (MPS) if margin < {thr:.4f}")

    if a.bench:
        from .bench import BENCH_N, REPEATS
        lat = np.zeros((REPEATS, BENCH_N))
        escs = np.zeros(BENCH_N, bool)
        for r in range(REPEATS):
            for i in range(BENCH_N):
                o = router(normalize(x_test[i:i + 1]))
                if o["escalated"]:
                    router.torch.mps.synchronize()
                lat[r, i] = o["latency_ms"]
                escs[i] = o["escalated"]
        per = np.median(lat, 0)
        res = {"threshold": thr, "frac_escalated": float(escs.mean()), "lat_mean_ms": float(per.mean()),
               "lat_p50_ms": float(np.percentile(per, 50)), "lat_p99_ms": float(np.percentile(per, 99))}
        (RESULTS_DIR / "routing_e2e.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    items = [(p, preprocess_file(p), None) for p in a.images]
    items += [(f"cifar-test[{i}]", normalize(x_test[i:i + 1]), int(y_test[i])) for i in a.cifar_test]
    for name, x, label in items:
        o = router(x)
        print(f"{name}: INT8={CIFAR10_CLASSES[o['int8_pred']]} margin={o['margin']:.3f} "
              f"{'ESCALATE->FP16' if o['escalated'] else 'keep INT8'} final={CIFAR10_CLASSES[o['final_pred']]} "
              f"latency={o['latency_ms']:.1f}ms" + (f" (true={CIFAR10_CLASSES[label]})" if label is not None else ""))


if __name__ == "__main__":
    main()
