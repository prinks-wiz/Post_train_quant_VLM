"""Reads the committed results/ files into one JSON-able dict for tabs 2-4 (stdlib only).

Nothing here is re-measured: every number comes from a file in results/ (or, for the two FP16
warm/idle probe numbers that were only recorded in prose, from the README's root-cause table).
Missing files become entries in `missing` instead of errors.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PLOTS = {
    "accuracy_vs_N": "results/accuracy_vs_N.png",
    "pareto_ptq": "results/pareto.png",
    "p99_vs_escalation": "results/stream/p99_vs_escalation.png",
    "pareto_lean": "results/lean/pareto_lean.png",
}

STREAM_VARIANTS = [("int8_only", "INT8 only (CPU)"), ("fp16_only", "FP16 only (GPU)"),
                   ("route_naive", "Naive routing"), ("route_keepwarm", "+ keep-warm"),
                   ("route_microbatch", "+ micro-batching"), ("route_deadline", "+ deadline-aware"),
                   ("route_all", "All fixes")]

LEAN_KEY = [("fp32_d12_aligned_r224", "FP32, 224 px, aligned"),
            ("int8_dyn_d12_aligned_r224__np", "INT8-dyn, 224 px"),
            ("int8_dyn_d12_aligned_r128-224_f20__np", "INT8-dyn, cascade 128 → 224"),
            ("int4_a4_d12_aligned_r224__np", "INT4(a4), 224 px"),
            ("int4_a4_d12_aligned_r128-224_f20__np", "INT4(a4), cascade 128 → 224"),
            ("int8_dyn_d12_probe_r128-224_f10__np", "Probe cascade 128 → 224 (labels)")]

STREAM_COMPARE = {"stream/route_naive (torch+MPS)": ("torch + MPS: INT8 → FP16 router", "ref"),
                  "stream/fp16_only (torch+MPS)": ("torch + MPS: FP16 every frame", "ref"),
                  "stream/int8_only (torch+MPS)": ("torch + MPS: INT8 only", "ref"),
                  "int8_dyn_d12_aligned_r224__np_nopack": ("lean INT8-dyn 224", "lean"),
                  "int8_dyn_d12_aligned_r128-224_f20__np": ("lean INT8-dyn cascade 128 → 224", "cascade"),
                  "int4_a4_d12_aligned_r128-224_f20__np": ("lean INT4(a4) cascade 128 → 224", "cascade"),
                  "int4_a4_d12_aligned_r224__np": ("lean INT4(a4) 224", "lean"),
                  "int8_dyn_d12_probe_r128-224_f10__np+deadline": ("lean probe cascade (labels)", "probe")}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _csv(rel, missing):
    p = ROOT / rel
    if not p.exists():
        missing.append(rel)
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _json(rel, missing):
    p = ROOT / rel
    if not p.exists():
        missing.append(rel)
        return None
    return json.loads(p.read_text())


def phase1(missing):
    rows = _csv("results/results.csv", missing)
    curve = {}
    for r in rows:
        if r.get("precision") != "int8":
            continue
        sel = "k-center" if "kcenter" in r["selection"] else "random"
        curve.setdefault(sel, {}).setdefault(int(r["N"]), []).append(100 * _f(r["top1_acc"]))
    fp32 = next((100 * _f(r["top1_acc"]) for r in rows if r["variant"] == "fp32_ort_cpu"), None)
    routing = [r for r in _csv("results/routing.csv", missing) if r.get("route") and r.get("target")]
    return {"curve": curve, "fp32": fp32,
            "auroc": _json("results/routing_auroc.json", missing),
            "e2e": _json("results/routing_e2e.json", missing),
            "routing": [{k: (_f(v) if k not in ("route", "target") else v) for k, v in r.items()}
                        for r in routing]}


def phase2(missing):
    rows = _csv("results/stream/stream_results.csv", missing)
    table = []
    for key, label in STREAM_VARIANTS:
        row = {"variant": label, "key": key}
        for src in ("fixed30", "fixed60", "bursty"):
            r = next((r for r in rows if r["variant"] == key and r["source"] == src
                      and abs(_f(r["esc_frac"]) - 0.1) < 1e-9), None)
            if r:
                row[src] = {"p50": _f(r["p50_ms"]), "p99": _f(r["p99_ms"]), "acc": 100 * _f(r["acc_all"]),
                            "esc": 100 * _f(r["esc_rate"]), "reps": r.get("reps")}
        table.append(row)
    # FP16 MPS warm vs after-idle probe: recorded in the README root-cause table only.
    warm_idle, readme = None, ROOT / "README.md"
    if readme.exists():
        m = re.search(r"\*\*([\d.]+) ms\*\* back-to-back; \*\*~([\d.]+) ms\*\* after", readme.read_text())
        if m:
            warm_idle = {"warm": float(m.group(1)), "idle": float(m.group(2)), "source": "README.md root-cause table"}
    if warm_idle is None:
        missing.append("README.md (FP16 warm/idle probe numbers)")
    return {"table": table, "warm_idle": warm_idle}


def phase3(missing):
    bench = {r["spec"]: r for r in _csv("results/lean/bench_results.csv", missing)
             if r.get("threads") == "-1" and r.get("lean_mem") == "True"}
    key = []
    for spec, label in LEAN_KEY:
        r = bench.get(spec)
        if r:
            key.append({"label": label, "spec": spec, "acc": 100 * _f(r["acc"]), "p50": _f(r["lat_p50_ms"]),
                        "p99": _f(r["lat_p99_ms"]), "mean": _f(r["lat_mean_ms"]),
                        "rss": _f(r["peak_rss_mb"]), "rerun": 100 * (_f(r["frac_stage_b"]) or 0),
                        "probe": "probe" in spec})
    online = []
    for r in _csv("results/lean/stream_compare.csv", missing):
        if r["source"] == "fixed30" and r["spec"] in STREAM_COMPARE:
            label, group = STREAM_COMPARE[r["spec"]]
            online.append({"label": label, "group": group, "spec": r["spec"], "p50": _f(r["p50_ms"]),
                           "p99": _f(r["p99_ms"]), "acc": _f(r["acc_all"]), "rss": _f(r["peak_rss_mb"]),
                           "rerun": _f(r["frac_stage_b"])})
    abl = []
    p = RES / "lean" / "alignment_ablation.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"(\S+)\s+r(\d+): original CLIP projection ([\d.]+) \| head fit on FP32 feats ([\d.]+)"
                         r" \| head fit on own \(quantized\) feats ([\d.]+)", line)
            if m:
                abl.append({"precision": m.group(1), "res": int(m.group(2)), "orig": float(m.group(3)),
                            "fp32_fit": float(m.group(4)), "own_fit": float(m.group(5))})
    else:
        missing.append("results/lean/alignment_ablation.txt")
    return {"key": key, "online": online, "ablation": abl}


def load():
    missing = []
    out = {"phase1": phase1(missing), "phase2": phase2(missing), "phase3": phase3(missing)}
    out["plots"] = {k: v for k, v in PLOTS.items() if (ROOT / v).exists()}
    missing += [v for v in PLOTS.values() if not (ROOT / v).exists()]
    out["missing"] = missing
    return out


if __name__ == "__main__":
    print(json.dumps(load(), indent=1)[:4000])
