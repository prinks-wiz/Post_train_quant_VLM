"""Build all static INT8 variants (selection x N)."""
import time

from .common import CALIB_SIZES, MODEL_DIR
from .data_cache import load, normalize
from .quantize import export_fp32, quantize
from .select import build_selections


def int8_path(selection: str, n: int):
    return MODEL_DIR / f"int8_{selection}_n{n}.onnx"


def build_all():
    export_fp32()
    orders = build_selections()
    pool, _ = load("pool")
    for name, order in orders.items():
        for n in CALIB_SIZES:
            out = int8_path(name, n)
            if out.exists():
                continue
            t = time.time()
            quantize(normalize(pool[sorted(order[:n])]), out)
            print(f"built {out.name} in {time.time() - t:.1f}s", flush=True)
