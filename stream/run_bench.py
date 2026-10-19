"""Streaming benchmark on the labeled CIFAR-10 replay stream.

    python -m stream.run_bench                 # main table (+ escalation sweep for the plot)
    python -m stream.run_bench --quick         # checkpoint subset: 30 fps, naive vs keep-warm
    python -m stream.run_bench --no-sweep      # skip the p99-vs-escalation sweep
    python -m stream.run_bench --report        # rebuild table/plot from saved runs

Each run is a fresh process (`--one '<json>'`) so GPU/CPU state from a previous variant can't leak.
Mine: variant grid, metrics (latency percentiles from capture to output, deadline misses, drops,
throughput, accuracy), aggregation and plot. Results -> results/stream/.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from ptq.common import RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "stream"
RUNS = OUT / "runs"
N_FRAMES = 1500
ESC_FRAC = 0.10

VARIANTS = {
    "int8_only":       dict(mode="int8"),
    "fp16_only":       dict(mode="fp16"),
    "route_naive":     dict(mode="route"),
    "route_keepwarm":  dict(mode="route", keep_warm=True),
    "route_microbatch": dict(mode="route", microbatch=True),
    "route_deadline":  dict(mode="route", deadline_aware=True),
    "route_all":       dict(mode="route", keep_warm=True, microbatch=True, deadline_aware=True),
}
SOURCES = {"fixed30": ("fixed", 30.0), "fixed60": ("fixed", 60.0), "bursty": ("bursty", 30.0)}
SWEEP_FRACS = (0.05, 0.20, 0.30)


def run_one(spec: dict) -> dict:
    """Execute one streaming run in this process; returns a summary + per-frame records."""
    from .pipeline import Config, Pipeline
    from .sources import play_replay, replay_preprocess
    from .thresholds import threshold_for
    from ptq.common import seed_everything
    from ptq.data_cache import load, normalize

    seed_everything(0)
    mode, fps = SOURCES[spec["source"]]
    deadline_ms = 1000.0 / fps
    cfg = Config(**VARIANTS[spec["variant"]], threshold=threshold_for(spec["esc_frac"]), deadline_ms=deadline_ms)
    pipe = Pipeline(cfg, replay_preprocess)
    x, _ = load("test")
    pipe.warmup(normalize(x[:1]))
    pipe.start()
    cap = play_replay(pipe, spec["n"], mode, fps)
    cap.join()
    pipe.close()

    recs = sorted(pipe.records, key=lambda f: f.fid)
    lat = np.array([(f.t_out - f.t_cap) * 1e3 for f in recs])
    pred = np.array([f.pred for f in recs])
    lab = np.array([f.label for f in recs])
    n_total = spec["n"]
    span = max(f.t_out for f in recs) - min(f.t_cap for f in recs + pipe.dropped)
    esc_lat = lat[[f.path == "fp16" for f in recs]]
    return {**spec, "threshold": cfg.threshold, "deadline_ms": deadline_ms,
            "n_frames": n_total, "n_done": len(recs), "dropped": len(pipe.dropped),
            "late": int((lat > deadline_ms).sum()),
            "miss_rate": float(((lat > deadline_ms).sum() + len(pipe.dropped)) / n_total),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)), "mean_ms": float(lat.mean()), "max_ms": float(lat.max()),
            "esc_p50_ms": float(np.percentile(esc_lat, 50)) if len(esc_lat) else None,
            "esc_p99_ms": float(np.percentile(esc_lat, 99)) if len(esc_lat) else None,
            "throughput_fps": len(recs) / span,
            "acc_done": float((pred == lab).mean()), "acc_all": float((pred == lab).sum() / n_total),
            "esc_rate": float(np.mean([f.path == "fp16" for f in recs])),
            "deadline_kept": int(sum(f.deadline_kept for f in recs)),
            "mean_gpu_batch": float(np.mean([f.extra.get("gpu_batch", 1) for f in recs if f.path == "fp16"]))
            if len(esc_lat) else None,
            "warm_runs": pipe.n_warm_runs}


def launch(spec: dict) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{spec['source']}__{spec['variant']}__esc{int(spec['esc_frac'] * 100)}.json"
    if not path.exists():
        r = subprocess.run([sys.executable, "-m", "stream.run_bench", "--one", json.dumps(spec)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        path.write_text(r.stdout.strip().splitlines()[-1])
    res = json.loads(path.read_text())
    print(f"{res['source']:8s} {res['variant']:17s} esc{int(res['esc_frac'] * 100):>2d}%  "
          f"p50 {res['p50_ms']:6.1f}  p99 {res['p99_ms']:6.1f}  acc {res['acc_all'] * 100:5.1f}  "
          f"esc {res['esc_rate'] * 100:4.1f}%  drop {res['dropped']}  late {res['late']}", flush=True)
    return res


def reference_accuracy(n):
    """Offline accuracy on the same replay subset, from the ptq/ cached embeddings (read-only)."""
    from ptq.common import logits_from_embeddings, text_embeddings
    from ptq.data_cache import load
    from ptq.evaluate import EMB_DIR
    from .sources import replay_subset
    _, y = load("test")
    idx = replay_subset(n)
    te = text_embeddings()
    return {k: float((logits_from_embeddings(np.load(EMB_DIR / f"{k}.npy")[idx], te).argmax(1) == y[idx]).mean())
            for k in ("fp32_ort_cpu", "fp16_torch_mps", "int8_kcenter_n512")}


def report(rows):
    from .plot import plot_p99_vs_esc
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stream_results.csv", index=False)
    ref = reference_accuracy(N_FRAMES)
    main = df[df.esc_frac == ESC_FRAC].copy()
    order = {v: i for i, v in enumerate(VARIANTS)}
    main = main.sort_values(["source", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s)
    lines = [f"Replay of {N_FRAMES} CIFAR-10 test images (seeded subset). Escalation threshold = 10% quantile "
             f"of INT8 margins on the train pool. Deadline = 1 frame interval (33.3 ms at 30 fps and bursty, "
             f"16.7 ms at 60 fps). Offline accuracy on this subset: FP32 {ref['fp32_ort_cpu'] * 100:.2f}%, "
             f"FP16 {ref['fp16_torch_mps'] * 100:.2f}%, INT8 {ref['int8_kcenter_n512'] * 100:.2f}%.\n"]
    for src, g in main.groupby("source", sort=False):
        t = g[["variant", "p50_ms", "p95_ms", "p99_ms", "esc_p99_ms", "throughput_fps", "dropped", "late",
               "miss_rate", "acc_all", "esc_rate", "deadline_kept", "mean_gpu_batch"]].copy()
        for c in ("miss_rate", "acc_all", "esc_rate"):
            t[c] = t[c] * 100
        lines.append(f"### {src}\n\n" + t.to_markdown(index=False, floatfmt=".1f") + "\n")
    (OUT / "stream_results.md").write_text("\n".join(lines))
    plot_p99_vs_esc(df, OUT / "p99_vs_escalation.png")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=N_FRAMES)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(run_one(json.loads(a.one))))
        return
    if a.report:
        report([json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))])
        return
    if a.quick:
        specs = [dict(source="fixed30", variant=v, esc_frac=ESC_FRAC, n=a.n)
                 for v in ("int8_only", "fp16_only", "route_naive", "route_keepwarm")]
    else:
        specs = [dict(source=s, variant=v, esc_frac=ESC_FRAC, n=a.n) for s in SOURCES for v in VARIANTS]
        if not a.no_sweep:
            specs += [dict(source="fixed30", variant=v, esc_frac=f, n=a.n)
                      for f in SWEEP_FRACS for v in VARIANTS if v.startswith("route")]
    rows = [launch(s) for s in specs]
    report(rows)


if __name__ == "__main__":
    main()
