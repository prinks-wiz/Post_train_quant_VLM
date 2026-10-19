"""Confidence routing: run INT8; if its top-1/top-2 softmax margin is below a threshold, escalate
the input to a higher-precision model.

- INT8 model is fixed in advance (k-center, N=512) - not picked by test accuracy.
- Escalation targets: FP16 on PyTorch MPS (plan's target; different device than INT8 CPU) and
  FP32 on ORT CPU (same-backend alternative).
- Accuracy uses the full 10k test set. Latency per image = measured batch-1 INT8 latency +
  (if escalated) measured batch-1 target latency for that same image, over the fixed benchmark
  images. `python -m ptq.demo --bench` validates this composition with a real end-to-end run.
- AUROC: how well low margin detects INT8-vs-FP32 top-1 disagreement. Also reports distance to the
  nearest calibration embedding as an alternative score.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CALIB_DIR, RESULTS_DIR, logits_from_embeddings, text_embeddings  # noqa: E402
from .data_cache import load  # noqa: E402
from .evaluate import BENCH_DIR, EMB_DIR, softmax  # noqa: E402
from .report import GRID, INK, KCENTER, MUTED, RANDOM, REF, _style  # noqa: E402
from .select import POOL_EMB  # noqa: E402

INT8 = "int8_kcenter_n512"
TARGETS = {"fp16_torch_mps": "→ FP16 (MPS)", "fp32_ort_cpu": "→ FP32 (ORT CPU)"}
ESC_FRACS = [0, .02, .05, .1, .15, .2, .3, .4, .5, .75, 1.0]


def margin(probs):
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def preds_probs(name, te):
    lg = logits_from_embeddings(np.load(EMB_DIR / f"{name}.npy"), te)
    return lg.argmax(1), softmax(lg)


def bench(name):
    return json.loads((BENCH_DIR / f"{name}.json").read_text())


def run_routing():
    _, y = load("test")
    te = text_embeddings()
    p8, pr8 = preds_probs(INT8, te)
    p32, _ = preds_probs("fp32_ort_cpu", te)
    m = margin(pr8)
    disagree = (p8 != p32).astype(int)

    # Alternative score: distance from the INT8 embedding to the nearest calibration image (FP32 emb).
    order = json.loads((CALIB_DIR / "orderings.json").read_text())["kcenter"][:512]
    cal = np.load(POOL_EMB)[order]
    e8 = np.load(EMB_DIR / f"{INT8}.npy")
    e8 /= np.linalg.norm(e8, axis=1, keepdims=True)
    dist = 1 - (e8 @ cal.T).max(1)
    auroc = {"margin": float(roc_auc_score(disagree, -m)),
             "calib_distance": float(roc_auc_score(disagree, dist)),
             "n_disagreements": int(disagree.sum()), "n": int(len(y))}

    lat8 = np.array(bench(INT8)["per_image_ms"])
    nb = len(lat8)
    rows = []
    for tname, label in TARGETS.items():
        pt, _ = preds_probs(tname, te)
        lat_t = np.array(bench(tname)["per_image_ms"])
        for f in ESC_FRACS:
            thr = np.quantile(m, f) if 0 < f < 1 else (-1.0 if f == 0 else 2.0)
            esc = m < thr
            final = np.where(esc, pt, p8)
            lat = lat8 + esc[:nb] * lat_t
            rows.append({"route": f"INT8 {label}", "target": tname, "threshold": float(thr),
                         "frac_escalated": float(esc.mean()), "top1_acc": float((final == y).mean()),
                         "agree_fp32": float((final == p32).mean()),
                         "lat_mean_ms": float(lat.mean()), "lat_p99_ms": float(np.percentile(lat, 99))})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "routing.csv", index=False)
    (RESULTS_DIR / "routing_auroc.json").write_text(json.dumps(auroc, indent=1))
    pareto(df)

    show = df.copy()
    for c in ("frac_escalated", "top1_acc", "agree_fp32"):
        show[c] = show[c] * 100
    md = ("## Confidence routing (INT8 = k-center N=512)\n\n"
          f"AUROC for detecting INT8≠FP32 top-1 ({auroc['n_disagreements']} of {auroc['n']} test images): "
          f"margin **{auroc['margin']:.3f}**, distance-to-calibration-set **{auroc['calib_distance']:.3f}**\n\n"
          + show.drop(columns="target").to_markdown(index=False, floatfmt=".3f") + "\n")
    (RESULTS_DIR / "routing.md").write_text(md)
    print(md)
    return df, auroc


def pareto(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for (tname, g), color in zip(df.groupby("target", sort=False), (KCENTER, RANDOM)):
        g = g.sort_values("frac_escalated")
        ax.plot(g.lat_mean_ms, g.top1_acc * 100, "-o", color=color, lw=2, ms=6, mec="white", mew=1.2,
                label=f"Routed INT8 {TARGETS[tname]} (points = % escalated)")
        for _, r in g.iterrows():
            if any(abs(r.frac_escalated - f) < 2e-3 for f in (0.1, 0.2, 0.5)):
                ax.annotate(f"{r.frac_escalated * 100:.0f}%", (r.lat_mean_ms, r.top1_acc * 100),
                            textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=MUTED)
    res = pd.read_csv(RESULTS_DIR / "results.csv").set_index("variant")
    pts = [("fp32_ort_cpu", "FP32 ORT CPU"), ("fp32_torch_mps", "FP32 MPS"),
           ("fp16_torch_mps", "FP16 MPS"), (INT8, "INT8 ORT CPU (k-center 512)")]
    for v, lab in pts:
        r = res.loc[v]
        ax.plot(r.lat_mean_ms, r.top1_acc * 100, "D", color=REF if "int8" not in v else INK, ms=7,
                mec="white", mew=1.2, zorder=5)
        ax.annotate(lab, (r.lat_mean_ms, r.top1_acc * 100), textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=INK)
    ax.set_xlabel("Mean batch-1 latency per image (ms)", color=MUTED, fontsize=9)
    ax.set_title("Accuracy vs latency: pure precisions and confidence-routed INT8", loc="left",
                 color=INK, fontsize=11)
    ax.set_ylabel("Top-1 accuracy (%)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pareto.png")
    plt.close(fig)
