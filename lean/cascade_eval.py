"""Offline cascade analysis from cached FP32 features (fast; no inference).

Cost proxy = tokens x layers (attention/MLP FLOPs scale ~linearly in both at these tiny token counts).
- depth cascade (same res): exit after k1 layers; if margin < thr continue to layer 12 (no recompute).
- resolution cascade (shared weights): full trunk at low res; if margin < thr re-run at 224.
Mine: all of it.
"""
import numpy as np

from .offline import feats, fit_ridge, raw_data
from .runtime import LEAN_ART, LOGIT_SCALE, TEXT_EMB, margin

FULL_COST = 50 * 12


def aligned_logits(res, depth, cache={}):
    key = (res, depth)
    if key not in cache:
        W, b, _ = fit_ridge(feats(res, "pool", depth), np.load(LEAN_ART / "target_pool.npy"))
        z = feats(res, "test", depth) @ W + b
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cache[key] = LOGIT_SCALE * z @ np.load(TEXT_EMB).T
    return cache[key]


def sweep(lg1, lg2, cost1, cost2, y, fracs=np.linspace(0, 1, 21)):
    m = margin(lg1)
    p1, p2 = lg1.argmax(1), lg2.argmax(1)
    rows = []
    for f in fracs:
        thr = np.quantile(m, f) if 0 < f < 1 else (-1 if f == 0 else 2)
        esc = m < thr
        rows.append({"frac_second": float(esc.mean()), "acc": float((np.where(esc, p2, p1) == y).mean()),
                     "rel_cost": float((cost1 + esc.mean() * cost2) / FULL_COST)})
    return rows


def run():
    *_, y = raw_data()
    out = {}
    for res, k1 in ((224, 8), (224, 10), (160, 10)):
        t = 1 + (res // 32) ** 2
        out[f"depth r{res} exit@{k1}->12"] = sweep(aligned_logits(res, k1), aligned_logits(res, 12),
                                                   t * k1, t * (12 - k1), y)
    for res in (64, 96, 128, 160):
        t = 1 + (res // 32) ** 2
        out[f"res r{res}->r224"] = sweep(aligned_logits(res, 12), aligned_logits(224, 12), t * 12, FULL_COST, y)
    out["res r128->r160"] = sweep(aligned_logits(128, 12), aligned_logits(160, 12), 17 * 12, 26 * 12, y)
    return out


if __name__ == "__main__":
    for name, rows in run().items():
        best = [r for r in rows if r["acc"] >= 0.875]
        b = min(best, key=lambda r: r["rel_cost"]) if best else None
        print(f"{name:24s} start acc {rows[0]['acc']:.4f} cost {rows[0]['rel_cost']:.2f} | "
              + (f"cheapest >=87.5%: acc {b['acc']:.4f} cost {b['rel_cost']:.2f} 2nd {b['frac_second']:.2f}" if b else "never >=87.5%")
              + f" | @30% 2nd: acc {rows[6]['acc']:.4f} cost {rows[6]['rel_cost']:.2f}")
