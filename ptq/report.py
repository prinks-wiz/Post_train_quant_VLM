"""Results table (CSV + markdown) and accuracy-vs-N plot."""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CALIB_DIR, RESULTS_DIR  # noqa: E402
from .evaluate import metrics  # noqa: E402

KCENTER, RANDOM, FP16, REF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(df: pd.DataFrame, path):
    int8 = df[df.precision == "int8"]
    km = int8[int8.selection == "kcenter"].sort_values("N")
    rnd = int8[int8.selection.str.startswith("random")].groupby("N")
    fp32 = df[df.variant == "fp32_ort_cpu"].iloc[0]
    fp16 = df[df.variant == "fp16_torch_mps"].iloc[0]
    ns = km.N.values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    for ax, col, title in [(axes[0], "top1_acc", "CIFAR-10 zero-shot top-1 accuracy (%)"),
                           (axes[1], "agree_fp32", "Top-1 agreement with FP32 (%)")]:
        _style(ax)
        r_mean, r_min, r_max = rnd[col].mean() * 100, rnd[col].min() * 100, rnd[col].max() * 100
        ax.fill_between(ns, r_min.values, r_max.values, color=RANDOM, alpha=0.15, linewidth=0)
        ax.plot(ns, r_mean.values, "-o", color=RANDOM, lw=2, ms=7, mec="white", mew=1.5,
                label="INT8, random (mean of 3 seeds; band = min–max)")
        ax.plot(ns, km[col].values * 100, "-o", color=KCENTER, lw=2, ms=7, mec="white", mew=1.5,
                label="INT8, k-center coverage")
        ax.axhline(fp32[col] * 100, color=REF, lw=1.5, ls="--", label="FP32 (ORT CPU)")
        if col == "top1_acc":
            ax.axhline(fp16[col] * 100, color=FP16, lw=1.5, ls=":", label="FP16 (PyTorch MPS)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("Calibration set size N (images from train split)", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        for x, yv in zip(ns, km[col].values * 100):
            ax.annotate(f"{yv:.1f}", (x, yv), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
        for x, yv in zip(ns, r_mean.values):
            ax.annotate(f"{yv:.1f}", (x, yv), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics())
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    plot(df, RESULTS_DIR / "accuracy_vs_N.png")

    show = df.copy()
    for c in ("top1_acc", "agree_fp32"):
        show[c] = (show[c] * 100).map("{:.2f}".format)
    for c in ("cos_fp32_mean", "cos_fp32_p01"):
        show[c] = show[c].map("{:.4f}".format)
    for c in ("lat_mean_ms", "lat_p50_ms", "lat_p99_ms"):
        show[c] = show[c].map("{:.2f}".format)
    show["peak_rss_mb"] = show["peak_rss_mb"].map("{:.0f}".format)
    show["model_file_mb"] = show["model_file_mb"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    cols = ["variant", "backend", "precision", "selection", "N", "top1_acc", "agree_fp32", "cos_fp32_mean",
            "cos_fp32_p01", "lat_p50_ms", "lat_p99_ms", "peak_rss_mb", "model_file_mb"]
    md = show[cols].to_markdown(index=False)

    # Random-vs-k-center summary per N (mean/min/max over random seeds).
    int8 = df[df.precision == "int8"]
    summ = []
    for n, g in int8.groupby("N"):
        r = g[g.selection.str.startswith("random")]
        k = g[g.selection == "kcenter"].iloc[0]
        summ.append({"N": n, "kcenter_acc": k.top1_acc * 100, "random_acc_mean": r.top1_acc.mean() * 100,
                     "random_acc_min": r.top1_acc.min() * 100, "random_acc_max": r.top1_acc.max() * 100,
                     "kcenter_agree": k.agree_fp32 * 100, "random_agree_mean": r.agree_fp32.mean() * 100,
                     "kcenter_cos": k.cos_fp32_mean, "random_cos_mean": r.cos_fp32_mean.mean()})
    summ = pd.DataFrame(summ)
    stats = pd.DataFrame(json.loads((CALIB_DIR / "selection_stats.json").read_text()))
    stats = stats.drop(columns="class_counts")
    (RESULTS_DIR / "results.md").write_text(
        "## All variants (full 10,000-image CIFAR-10 test set)\n\n" + md +
        "\n\n## INT8: k-center vs random by N\n\n" + summ.to_markdown(index=False, floatfmt=".2f") +
        "\n\n## Calibration-set coverage (pool of 5,000 train images)\n\n" +
        stats.to_markdown(index=False, floatfmt=".3f") + "\n")
    print((RESULTS_DIR / "results.md").read_text())
    return df
