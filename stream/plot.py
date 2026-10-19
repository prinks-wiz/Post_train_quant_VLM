"""p99 end-to-end latency vs realized escalation rate, one line per routing variant (30 fps replay)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ptq.report import GRID, INK, MUTED, _style  # noqa: E402

# Categorical slots 1-4 of the reference palette; violet (slot 7) for the combined variant.
COLORS = {"route_naive": "#52514e", "route_keepwarm": "#2a78d6", "route_microbatch": "#eb6834",
          "route_deadline": "#1baf7a", "route_all": "#6250d6"}
LABELS = {"route_naive": "naive routing", "route_keepwarm": "+ keep-warm", "route_microbatch": "+ micro-batching",
          "route_deadline": "+ deadline-aware", "route_all": "all fixes"}


def plot_p99_vs_esc(df, path):
    d = df[df.source == "fixed30"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for v, color in COLORS.items():
        g = d[d.variant == v].sort_values("esc_rate")
        if g.empty:
            continue
        ax.plot(g.esc_rate * 100, g.p99_ms, "-o", color=color, lw=2, ms=7, mec="white", mew=1.5,
                label=LABELS[v])
        last = g.iloc[-1]
        ax.annotate(LABELS[v], (last.esc_rate * 100, last.p99_ms), textcoords="offset points",
                    xytext=(8, -3), fontsize=8, color=INK)
    for v, ls, lab in (("int8_only", "--", "INT8 only"), ("fp16_only", ":", "FP16 only (GPU)")):
        r = d[d.variant == v]
        if not r.empty:
            ax.axhline(r.p99_ms.iloc[0], color=MUTED, lw=1.2, ls=ls)
            ax.annotate(lab, (0, r.p99_ms.iloc[0]), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(4, 3), fontsize=8, color=MUTED)
    ax.axhline(1000 / 30, color="#e34948", lw=1, alpha=0.6)
    ax.annotate("deadline 33.3 ms", (1, 1000 / 30), xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(-4, 3), ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("Realized escalation rate (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("p99 end-to-end latency (ms)", color=MUTED, fontsize=9)
    ax.set_title("Streaming p99 latency vs escalation rate (30 fps CIFAR-10 replay)", loc="left",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
