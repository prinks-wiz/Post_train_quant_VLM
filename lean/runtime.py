"""Deploy-time code for the lean CPU-only path. Imports ONLY numpy, OpenCV and onnxruntime (no torch,
no transformers, no pandas) so the measured process RAM is what a deployment would actually pay.

Mine: preprocessing at reduced resolution, the aligned linear head, the early-exit cascade logic.
Library: onnxruntime inference, OpenCV resize.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEAN_ART = ROOT / "artifacts" / "lean"
TEXT_EMB = ROOT / "artifacts" / "text_emb.npy"     # frozen FP32 CLIP text embeddings (ptq/, read-only)

CLIP_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), np.float32).reshape(1, 3, 1, 1)
CLIP_STD = np.array((0.26862954, 0.26130258, 0.27577711), np.float32).reshape(1, 3, 1, 1)
LOGIT_SCALE = 100.0


_RESIZE = {}


def resize_matrix(res: int, src: int = 32) -> np.ndarray:
    """(res, src) matrix M with cv2.resize(INTER_CUBIC)(img) == M @ img @ M.T (bicubic is linear and
    separable). Built offline by resizing basis images with OpenCV; loaded here without OpenCV."""
    if res not in _RESIZE:
        _RESIZE[res] = np.load(LEAN_ART / "resize" / f"bicubic_{src}to{res}.npy")
    return _RESIZE[res]


def preprocess_np(raw_hwc_uint8: np.ndarray, res: int) -> np.ndarray:
    """OpenCV-free preprocessing: two small matmuls per image (+ the uint8 round/clip OpenCV applies)."""
    raw = raw_hwc_uint8[None] if raw_hwc_uint8.ndim == 3 else raw_hwc_uint8
    x = raw.astype(np.float32)
    if raw.shape[1] != res:
        M = resize_matrix(res, raw.shape[1])
        x = np.clip(np.rint(np.einsum("rh,bhwc,sw->bcrs", M, x, M, optimize=True)), 0, 255)
    else:
        x = x.transpose(0, 3, 1, 2)
    return (x / 255.0 - CLIP_MEAN) / CLIP_STD


def preprocess(raw_hwc_uint8: np.ndarray, res: int) -> np.ndarray:
    """(B,H,W,3) or (H,W,3) RGB uint8 -> (B,3,res,res) float32, CLIP-normalized. Bicubic resize."""
    import cv2
    raw = raw_hwc_uint8[None] if raw_hwc_uint8.ndim == 3 else raw_hwc_uint8
    out = np.empty((len(raw), 3, res, res), np.float32)
    for i, im in enumerate(raw):
        r = im if im.shape[0] == res else cv2.resize(im, (res, res), interpolation=cv2.INTER_CUBIC)
        out[i] = r.transpose(2, 0, 1)
    return (out / 255.0 - CLIP_MEAN) / CLIP_STD


class Head:
    """Head on 768-d trunk features.
    kind="aligned" (zero-shot, label-free): z = feat @ W + b maps the features into the full CLIP image
      embedding space (ridge fit to FP32 full-model embeddings of unlabeled train images), then cosine
      similarity with the frozen text embeddings.
    kind="probe" (USES LABELS - not zero-shot): logits = feat @ W + b (logistic regression on the
      5,000 labeled train-pool images)."""

    def __init__(self, path):
        d = np.load(path)
        self.W, self.b = d["W"].astype(np.float32), d["b"].astype(np.float32)
        self.kind = str(d["kind"]) if "kind" in d else "aligned"
        self.T = np.load(TEXT_EMB).astype(np.float32) if self.kind == "aligned" else None

    def logits(self, feat):
        z = feat @ self.W + self.b
        if self.kind == "probe":
            return z
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return LOGIT_SCALE * z @ self.T.T


def margin(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    top2 = np.sort(p, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def perf_cores() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                  text=True, check=True).stdout)
    except Exception:
        return 0


def session(path, threads: int = -1, lean_mem: bool = True, config: dict | None = None):
    """ORT CPU session. lean_mem disables the CPU memory arena and memory-pattern preallocation
    (lower RSS; small latency cost - measured, not assumed)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    t = perf_cores() if threads == -1 else threads
    if t:
        so.intra_op_num_threads = t
    so.inter_op_num_threads = 1
    if lean_mem:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    for k, v in (config or {}).items():   # e.g. {"session.disable_prepacking": "1"}
        so.add_session_config_entry(k, v)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Model:
    """One ONNX trunk session (weights loaded once) run at one or more resolutions.

    The trunk takes the position-embedding table as an input (`pos`, (1, 1+(res/32)^2, 768)), so the
    same weights serve any resolution. spec["stages"] = [res1, res2, ...]: run at res1; while the
    head's top1-top2 softmax margin is below spec["threshold"], re-run at the next resolution
    (resolution cascade). One aligned/probe head per resolution."""

    def __init__(self, spec: dict, threads: int = -1, lean_mem: bool = True):
        self.spec = spec
        self.sess = session(LEAN_ART / spec["onnx"], threads, lean_mem, spec.get("ort_config"))
        self.prep = preprocess_np if spec.get("prep") == "numpy" else preprocess
        self.stages = spec["stages"]
        self.pos = {r: np.load(LEAN_ART / "pos" / f"pos_r{r}.npy") for r in self.stages}
        self.heads = {r: Head(LEAN_ART / spec["heads"][str(r)]) for r in self.stages}
        self.threshold = spec.get("threshold", -1.0)

    def run_stage(self, raw, res):
        feat = self.sess.run(None, {"pixel_values": self.prep(raw, res), "pos": self.pos[res]})[0]
        return self.heads[res].logits(feat)

    def __call__(self, raw, allow_second=None):
        """Returns (logits, escalated: bool) for a single frame. `allow_second()` (optional) can veto the
        second pass, e.g. when it would miss the frame deadline (stream-aware cascade)."""
        lg = self.run_stage(raw, self.stages[0])
        for res in self.stages[1:]:
            if margin(lg)[0] >= self.threshold or (allow_second is not None and not allow_second()):
                return lg, False
            lg = self.run_stage(raw, res)
            return lg, True
        return lg, False


def load_spec(name):
    return json.loads((LEAN_ART / "specs" / f"{name}.json").read_text())
