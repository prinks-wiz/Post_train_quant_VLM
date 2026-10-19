"""Tables + Pareto plots for the lean experiments (results/lean/)."""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ptq.report import GRID, INK, MUTED, _style  # noqa: E402

from .runtime import ROOT  # noqa: E402

OUT = ROOT / "results" / "lean"
RUNS = OUT / "runs"
PREC_STYLE = {"fp32": ("#52514e", "s"), "int8_dyn": ("#eb6834", "^"), "int8_static": ("#2a78d6", "o"),
              "int4": ("#1baf7a", "D"), "int4_a4": ("#6250d6", "v")}


CALLOUTS = [("int8_dyn_d12_aligned_r224__np", "INT8-dyn 224", (14, 18)),
            ("int8_dyn_d12_aligned_r128-224_f20__np", "INT8-dyn 128→224", (-8, 34)),
            ("int4_a4_d12_aligned_r224__np", "INT4(a4) 224", (14, -16)),
            ("int4_a4_d12_aligned_r128-224_f20__np", "INT4(a4) 128→224", (6, -34)),
            ("int8_dyn_d12_aligned_r128", "INT8-dyn 128", (10, -14)),
            ("int8_dyn_d12_probe_r128-224_f10__np", "probe 128→224 (labels)", (12, -4)),
            ("fp32_d12_aligned_r224", "FP32 224", (-20, 14))]


def load(sub):
    rows = [json.loads(p.read_text()) for p in sorted((RUNS / sub).glob("*.json"))]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def bench_table():
    b = load("bench")
    acc = pd.read_csv(OUT / "accuracy.csv")
    g = b.groupby(["spec", "threads", "lean_mem"]).median(numeric_only=True).reset_index()
    return g.merge(acc, left_on="spec", right_on="name", how="left")


def pareto_plot(t, path):
    main = t[(t.threads == -1) & t.lean_mem]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    for ax, xcol, xlabel in ((axes[0], "lat_mean_ms", "Mean batch-1 latency, CPU, incl. preprocessing (ms)"),
                             (axes[1], "peak_rss_mb", "Peak process RSS (MB)")):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for prec, (color, marker) in PREC_STYLE.items():
            for kind, fill in (("aligned", color), ("probe", "white")):
                d = main[(main.precision == prec) & (main.head_kind == kind)]
                if d.empty:
                    continue
                ax.scatter(d[xcol], d.acc * 100, s=42, marker=marker, facecolor=fill, edgecolor=color,
                           linewidth=1.5, zorder=3,
                           label=f"{prec} {'zero-shot (aligned)' if kind == 'aligned' else 'probe (uses labels)'}")
        # zero-shot Pareto front (best accuracy so far as cost increases) as a step line
        zs = main[main.head_kind == "aligned"].sort_values(xcol)
        front = zs[zs.acc >= zs.acc.cummax()]
        ax.step(front[xcol], front.acc * 100, where="post", color=MUTED, lw=1, ls="--", zorder=1,
                label="zero-shot Pareto front")
        for spec, text, off in CALLOUTS:
            r = main[main.spec == spec]
            if not r.empty:
                ax.annotate(text, (r[xcol].iloc[0], r.acc.iloc[0] * 100), textcoords="offset points", xytext=off,
                            fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel("CIFAR-10 top-1 (%)", color=MUTED, fontsize=9)
    axes[0].set_title("Accuracy vs latency (lean CPU path)", loc="left", color=INK, fontsize=11)
    axes[1].set_title("Accuracy vs peak RAM", loc="left", color=INK, fontsize=11)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report():
    t = bench_table()
    t.to_csv(OUT / "bench_results.csv", index=False)
    pareto_plot(t, OUT / "pareto_lean.png")
    cols = ["spec", "precision", "head_kind", "acc", "frac_stage_b", "lat_p50_ms", "lat_p99_ms", "lat_mean_ms",
            "model_ram_mb", "peak_rss_mb", "rss_base_mb", "model_file_mb"]
    main = t[(t.threads == -1) & t.lean_mem].sort_values(["head_kind", "precision", "lat_mean_ms"])[cols].copy()
    main["acc"] *= 100
    main["frac_stage_b"] *= 100
    from .run_bench import ABLATE
    abl = t[t.spec.isin(ABLATE)][
        ["spec", "threads", "lean_mem", "lat_p50_ms", "lat_p99_ms", "model_ram_mb", "peak_rss_mb"]]
    r2 = main[main.spec.str.contains("__np") | main.spec.str.startswith("int4_a4")]
    main = main[~main.spec.isin(r2.spec)]
    md = ["## Lean CPU path: every variant (batch-1, 3 repeats, median; full-test accuracy)\n",
          main.to_markdown(index=False, floatfmt=".1f"),
          "\n## Round 2 (RAM): INT4 with INT8 compute (int4_a4), OpenCV-free preprocessing (__np), "
          "ORT weight prepacking off (__np_nopack)\n",
          r2.to_markdown(index=False, floatfmt=".1f"),
          "\n## Ablations: intra-op threads (-1 = 6 P-cores) and ORT memory arena (lean_mem=False = arena on)\n",
          abl.to_markdown(index=False, floatfmt=".1f")]
    s, r = load("stream"), load("router")
    if not s.empty:
        sg = s.groupby(["spec", "source"]).median(numeric_only=True).reset_index()
        cols_s = ["spec", "source", "p50_ms", "p95_ms", "p99_ms", "miss_rate", "acc_all", "frac_stage_b",
                  "peak_rss_mb"]
        if "second_pass_vetoed" not in sg:
            sg["second_pass_vetoed"] = np.nan
        st = sg[cols_s].copy()
        if not r.empty:
            rg = r.groupby(["variant", "source"]).median(numeric_only=True).reset_index()
            rg = rg.rename(columns={"variant": "spec", "esc_rate": "frac_stage_b"})
            rg["spec"] = "stream/" + rg["spec"] + " (torch+MPS)"
            st = pd.concat([st, rg[cols_s]], ignore_index=True)
        for c in ("miss_rate", "acc_all", "frac_stage_b"):
            st[c] *= 100
        st = st.sort_values(["source", "peak_rss_mb"])
        md += ["\n## Online replay (600 frames, 3 repeats, median): lean finalists vs the INT8->FP16 router\n",
               st.to_markdown(index=False, floatfmt=".1f")]
        st.to_csv(OUT / "stream_compare.csv", index=False)
    (OUT / "lean_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
