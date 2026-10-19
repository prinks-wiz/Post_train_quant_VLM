"""Streaming benchmark on the labeled CIFAR-10 replay stream.

    python -m stream.run_bench                 # main table (+ escalation sweep for the plot)
    python -m stream.run_bench --quick         # checkpoint subset: 30 fps, naive vs keep-warm
    python -m stream.run_bench --no-sweep      # skip the p99-vs-escalation sweep
    python -m stream.run_bench --report        # rebuild table/plot from saved runs

Each run is a fresh process (`--one '<json>'`) so GPU/CPU state from a previous variant can't leak.
The whole benchmark holds a `caffeinate -dims` assertion: with the display asleep, macOS throttles
timers/scheduling and tail latencies became meaningless (observed: INT8-only p99 278 ms vs 20 ms).
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
N_FRAMES = 600
ESC_FRAC = 0.10
REPS = 5          # main grid repeated, interleaved (rep is the outer loop); table = median over clean reps
CANARY_TOL = 1.25  # a run is "clean" if its pre/post canaries are within 25% of the all-run median

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
    x1 = normalize(x[:1])
    pipe.warmup(x1)
    pre = canary(pipe, x1)
    pipe.start()
    cap = play_replay(pipe, spec["n"], mode, fps)
    cap.join()
    pipe.close()
    post = canary(pipe, x1)

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
            "warm_runs": pipe.n_warm_runs, "ort_threads": pipe.ort_threads,
            "int8_run_p50_ms": _stage_p50(recs, "t_i0", "t_i1"), "gpu_run_p50_ms": _stage_p50(recs, "t_g0", "t_g1"),
            "canary_int8_ms": max(pre[0], post[0]), "canary_fp16_ms": max(pre[1], post[1])}


def canary(pipe, x1):
    """Machine-health probe independent of the variant: median isolated batch-1 INT8 and warm FP16
    latency. Background load / throttling on this shared laptop shows up here."""
    import time
    def med(fn, n):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn(x1)
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts))
    return med(pipe._run_int8, 40), med(pipe._run_fp16, 20)


def _stage_p50(recs, a, b):
    v = [(f.extra[b] - f.extra[a]) * 1e3 for f in recs if a in f.extra and b in f.extra]
    return float(np.median(v)) if v else None


def launch(spec: dict) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{spec['source']}__{spec['variant']}__esc{int(spec['esc_frac'] * 100)}__rep{spec['rep']}.json"
    if not path.exists():
        r = subprocess.run([sys.executable, "-m", "stream.run_bench", "--one", json.dumps(spec)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        path.write_text(r.stdout.strip().splitlines()[-1])
    res = json.loads(path.read_text())
    print(f"rep{res['rep']} {res['source']:8s} {res['variant']:17s} esc{int(res['esc_frac'] * 100):>2d}%  "
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


def mark_clean(df):
    ok8 = df.canary_int8_ms <= CANARY_TOL * df.canary_int8_ms.median()
    ok16 = df.canary_fp16_ms <= CANARY_TOL * df.canary_fp16_ms.median()
    return df.assign(clean=ok8 & ok16)


def aggregate(df):
    """Median over clean repeats (all repeats if none is clean) of every numeric column, plus the
    p99 range and the clean/total repeat counts."""
    keys = ["source", "variant", "esc_frac"]
    g = df.groupby(keys, sort=False)
    df = df.assign(reps_total=g.rep.transform("count"))
    df = df[df.clean | ~g.clean.transform("any")]
    num = [c for c in df.columns if c not in keys + ["rep", "reps_total", "clean"]
           and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(keys, sort=False)[num].median().reset_index()
    rng = df.groupby(keys, sort=False).agg(reps=("rep", "count"), reps_total=("reps_total", "first"),
                                          p99_min=("p99_ms", "min"),
                                          p99_max=("p99_ms", "max")).reset_index()
    return agg.merge(rng, on=keys)


def report(rows):
    from .plot import plot_p99_vs_esc
    raw = mark_clean(pd.DataFrame(rows))
    raw.to_csv(OUT / "stream_runs_raw.csv", index=False)
    df = aggregate(raw)
    df.to_csv(OUT / "stream_results.csv", index=False)
    n = int(raw.n.iloc[0])
    ref = reference_accuracy(n)
    main = df[df.esc_frac == ESC_FRAC].copy()
    order = {v: i for i, v in enumerate(VARIANTS)}
    main = main.sort_values(["source", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s)
    lines = [f"Replay of {n} CIFAR-10 test images (seeded subset); median over *clean* interleaved repeats "
             f"(`reps` = clean/total; clean = pre/post machine-health canary within {CANARY_TOL:.2f}x of the "
             f"all-run median; p99 range over the clean repeats). Escalation threshold = 10% quantile "
             f"of INT8 margins on the train pool. Deadline = 1 frame interval (33.3 ms at 30 fps and bursty, "
             f"16.7 ms at 60 fps). Offline accuracy on this subset: FP32 {ref['fp32_ort_cpu'] * 100:.2f}%, "
             f"FP16 {ref['fp16_torch_mps'] * 100:.2f}%, INT8 {ref['int8_kcenter_n512'] * 100:.2f}%.\n"]
    for src, g in main.groupby("source", sort=False):
        g = g.assign(reps=g.reps.astype(int).astype(str) + "/" + g.reps_total.astype(int).astype(str))
        t = g[["variant", "reps", "p50_ms", "p95_ms", "p99_ms", "p99_min", "p99_max", "esc_p99_ms", "throughput_fps", "dropped", "late",
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
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(run_one(json.loads(a.one))))
        return
    if a.report:
        report([json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))])
        return
    import os
    awake = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    try:
        if a.quick:
            specs = [dict(source="fixed30", variant=v, esc_frac=ESC_FRAC, n=a.n, rep=r) for r in range(a.reps)
                     for v in ("int8_only", "fp16_only", "route_naive", "route_keepwarm")]
        else:
            specs = [dict(source=s, variant=v, esc_frac=ESC_FRAC, n=a.n, rep=r)
                     for r in range(a.reps) for s in SOURCES for v in VARIANTS]
            if not a.no_sweep:  # sweep points: single repeat (the 10% point has all repeats)
                specs += [dict(source="fixed30", variant=v, esc_frac=f, n=a.n, rep=0)
                          for f in SWEEP_FRACS for v in VARIANTS if v.startswith("route")]
        rows = [launch(s) for s in specs]
        rows = [r for r in rows if r["n"] == a.n]
    finally:
        awake.terminate()
    report(rows)


if __name__ == "__main__":
    main()
