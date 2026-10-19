"""Build the static GitHub Pages version: `python -m demo.build_static [--out demo/_site] [--frames 450]`.

Tabs 2-4 are the same page fed the same results/ data. Tab 1 cannot run inference on a static host,
so this records a real session from the lean worker (CIFAR replay, cascade, 30 fps) and the page
replays it at the original timing, clearly labelled "recorded, not live". Stop `demo.app` first so
the recording isn't sharing the CPU with another worker.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time

from demo import offline
from demo.configs import CIFAR10_CLASSES, CONFIGS, DEFAULT

ROOT = offline.ROOT
STATIC = ROOT / "demo" / "static"


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        cpu = ""
    return cpu or platform.machine()


def record(config, fps, n):
    import psutil
    p = subprocess.Popen([sys.executable, "-m", "demo.worker", CONFIGS[config]["spec"]], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready, frames = None, []
    for raw in p.stdout:
        ev = json.loads(raw)
        if ev.get("state") == "ready":
            ready = ev
            break
    if ready is None:
        raise SystemExit("worker failed to start; run python -m demo.check")
    p.stdin.write(json.dumps({"cmd": "set", "source": "replay", "fps": fps, "deadline": False}) + "\n")
    p.stdin.flush()
    proc = psutil.Process(p.pid)
    for raw in p.stdout:
        ev = json.loads(raw)
        if ev.get("type") != "frame":
            continue
        ev["rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
        for k in ("lat_ms", "margin1", "peak_rss_mb"):
            ev[k] = round(ev[k], 3)
        ev["top3"] = [[c, round(q, 4)] for c, q in ev["top3"]]
        frames.append(ev)
        if len(frames) >= n:
            break
    p.stdin.write('{"cmd": "quit"}\n')
    p.stdin.flush()
    p.wait(3)
    return {"config": config, "fps": fps, "threshold": ready.get("threshold"), "frames": frames,
            "recorded": time.strftime("%Y-%m-%d"), "machine": machine()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo" / "_site"))
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-record", action="store_true", help="static tabs only; Live tab shows run-locally notice")
    a = ap.parse_args()

    out = offline.Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    data = offline.load()
    for rel in data["plots"].values():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, out / rel)
    rec = None
    if not a.no_record:
        print(f"recording {a.frames} live frames ({CONFIGS[DEFAULT]['label']}, {a.fps} fps)...")
        rec = record(DEFAULT, a.fps, a.frames)
        lat = sorted(f["lat_ms"] for f in rec["frames"])
        print(f"  p50 {lat[len(lat) // 2]:.1f} ms · p99 {lat[int(0.99 * (len(lat) - 1))]:.1f} ms · "
              f"escalated {sum(f['escalated'] for f in rec['frames']) / len(lat):.0%}")
    payload = {"info": {"configs": CONFIGS, "default": DEFAULT, "classes": CIFAR10_CLASSES, "webcam": False,
                        "mode": "static"},
               "offline": data, "recording": rec}
    (out / "data.js").write_text("window.DEMO_STATIC = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    html = (STATIC / "index.html").read_text()
    html = html.replace("</head>", '<script src="data.js"></script>\n</head>', 1)
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")
    print(f"wrote {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
