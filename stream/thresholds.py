"""Escalation thresholds chosen on held-out data: INT8 margins over the 5,000-image *train*
calibration pool (no test-set tuning). threshold(f) = f-quantile of pool margins."""
import numpy as np

from ptq.common import ART_DIR, logits_from_embeddings, text_embeddings
from ptq.data_cache import load, normalize

from .pipeline import _softmax_margin

MARGINS = ART_DIR / "stream" / "pool_int8_margins.npy"


def pool_margins() -> np.ndarray:
    if MARGINS.exists():
        return np.load(MARGINS)
    import onnxruntime as ort
    from ptq.build import int8_path
    sess = ort.InferenceSession(str(int8_path("kcenter", 512)), providers=["CPUExecutionProvider"])
    x, _ = load("pool")
    te = text_embeddings()
    m = np.concatenate([_softmax_margin(logits_from_embeddings(
        sess.run(None, {"pixel_values": normalize(x[i:i + 64])})[0], te)) for i in range(0, len(x), 64)])
    MARGINS.parent.mkdir(parents=True, exist_ok=True)
    np.save(MARGINS, m)
    return m


def threshold_for(frac: float) -> float:
    return float(np.quantile(pool_margins(), frac))
