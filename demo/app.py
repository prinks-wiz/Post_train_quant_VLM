"""Interview demo server: `python -m demo.app [--port 7860] [--no-webcam] [--no-browser]`.

Stdlib HTTP server + one HTML page. Inference runs in a separate `demo.worker` subprocess that
imports only numpy + onnxruntime, so the RSS shown is that worker's, not this server's.
Switching config restarts the worker, so the RSS readout always reflects exactly one config.

Mine: worker supervision, rolling live metrics, API. Library: http.server, psutil (RSS of the worker).
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo import offline
from demo.configs import CIFAR10_CLASSES, CONFIGS, DEFAULT

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
WINDOW = 300            # rolling window (frames) for p50/p99/escalation rate


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Worker:
    """Owns the inference subprocess and the rolling live metrics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.config = None
        self.ctl = {"source": "replay", "fps": 30, "deadline": False}
        self.reset()

    def reset(self):
        self.status = {"state": "starting", "msg": "starting worker"}
        self.lat = deque(maxlen=WINDOW)
        self.esc = deque(maxlen=WINDOW)
        self.correct = deque(maxlen=WINDOW)
        self.t_out = deque(maxlen=120)
        self.events = deque(maxlen=12)       # recent escalations / vetoes
        self.n = self.n_veto = self.n_esc = 0
        self.last = None
        self.drops = 0
        self.peak_rss = None

    def start(self, config):
        self.stop()
        with self.lock:
            self.reset()
            self.config = config
            self.t_start = time.time()
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "demo.worker", CONFIGS[config]["spec"]], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._read_err, args=(self.proc,), daemon=True).start()

    def stop(self):
        p = self.proc
        if p and p.poll() is None:
            try:
                p.stdin.write('{"cmd": "quit"}\n')
                p.stdin.flush()
                p.wait(2)
            except Exception:
                p.kill()

    def send(self, obj):
        p = self.proc
        if p and p.poll() is None:
            try:
                p.stdin.write(json.dumps(obj) + "\n")
                p.stdin.flush()
                return True
            except (BrokenPipeError, OSError):
                pass
        return False

    def control(self, ctl):
        self.ctl.update({k: ctl[k] for k in ("source", "fps", "deadline") if k in ctl})
        if self.status.get("state") == "ready":
            self.send({"cmd": "set", **self.ctl})

    def _read_err(self, proc):
        tail = deque(maxlen=20)
        for line in proc.stderr:
            tail.append(line.rstrip())
        if proc is self.proc and self.status.get("state") != "ready":
            self.status = {"state": "error", "msg": "worker exited: " + " | ".join(list(tail)[-3:])}

    def _read(self, proc):
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if proc is not self.proc:
                return
            with self.lock:
                if ev["type"] == "status":
                    ev["since_launch_s"] = time.time() - self.t_start
                    self.status = ev
                    self.peak_rss = ev.get("peak_rss_mb", self.peak_rss)
                    if ev["state"] == "ready":
                        self.send({"cmd": "set", **self.ctl})
                    continue
                self.n += 1
                self.lat.append(ev["lat_ms"])
                self.esc.append(ev["escalated"])
                self.n_esc += ev["escalated"]
                if ev["label"] is not None:
                    self.correct.append(ev["pred"] == ev["label"])
                self.t_out.append(time.time())
                self.drops = ev["drops"]
                self.peak_rss = ev["peak_rss_mb"]
                if ev["vetoed"]:
                    self.n_veto += 1
                if ev["vetoed"] or ev["escalated"]:
                    self.events.appendleft({"n": self.n, "vetoed": ev["vetoed"], "margin": ev["margin1"],
                                            "ms": round(ev["lat_ms"], 1), "t": time.strftime("%H:%M:%S")})
                self.last = ev
        if proc is self.proc and self.status.get("state") == "ready":
            self.status = {"state": "error", "msg": "worker stopped unexpectedly"}

    def rss_mb(self):
        p = self.proc
        if not p or p.poll() is not None:
            return None
        try:
            import psutil
            return psutil.Process(p.pid).memory_info().rss / 2**20
        except Exception:
            try:
                out = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True)
                return int(out.stdout.strip()) / 1024
            except Exception:
                return None

    def state(self):
        with self.lock:
            now = time.time()
            recent = [t for t in self.t_out if now - t < 2.0]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0.0
            lat = list(self.lat)
            return {
                "config": self.config, "status": self.status, "ctl": self.ctl, "last": self.last,
                "stats": {"n": self.n, "window": len(lat), "p50": pct(lat, 0.5), "p99": pct(lat, 0.99),
                          "fps": fps, "esc_rate": (sum(self.esc) / len(self.esc)) if self.esc else None,
                          "acc": (sum(self.correct) / len(self.correct)) if self.correct else None,
                          "acc_n": len(self.correct), "vetoes": self.n_veto, "escalations": self.n_esc,
                          "drops": self.drops, "rss_mb": self.rss_mb(), "peak_rss_mb": self.peak_rss,
                          "worker_pid": self.proc.pid if self.proc else None},
                "events": list(self.events),
            }


def make_handler(worker, info):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/info":
                self._json(info)
            elif path == "/api/offline":
                self._json(offline.load())
            elif path == "/api/state":
                self._json(worker.state())
            elif path.startswith("/results/") and path.endswith(".png"):
                super().do_GET()                      # committed plots only
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False}, 400)
            if self.path == "/api/config":
                cfg = body.get("config")
                if cfg not in CONFIGS:
                    return self._json({"ok": False, "error": "unknown config"}, 400)
                worker.start(cfg)
                self._json({"ok": True})
            elif self.path == "/api/control":
                worker.control(body)
                self._json({"ok": True})
            elif self.path == "/api/frame":
                self._json({"ok": worker.send({"cmd": "frame", "rgb": body.get("rgb", "")})})
            elif self.path == "/api/reset":
                with worker.lock:
                    cfg = worker.config
                worker.start(cfg)
                self._json({"ok": True})
            else:
                self.send_error(404)
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-webcam", action="store_true", help="hide the webcam source")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument("--config", default=DEFAULT, choices=list(CONFIGS))
    a = ap.parse_args()

    worker = Worker()
    worker.start(a.config)          # loads + warms the model while the page opens
    info = {"configs": CONFIGS, "default": a.config, "classes": CIFAR10_CLASSES,
            "webcam": not a.no_webcam, "mode": "live"}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(worker, info))
    url = f"http://{a.host}:{a.port}/"
    print(f"demo: {url}   (Ctrl+C to quit)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
