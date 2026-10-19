"""End-to-end pipeline (core experiment + confidence routing). Every stage caches to artifacts/ and is skipped if already done.

    python -m ptq.run_all            # full core experiment + routing
    python -m ptq.run_all --report   # only regenerate tables/plots from cached artifacts
"""
import argparse
import time

from .common import seed_everything, text_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    seed_everything()
    t0 = time.time()
    if not a.report:
        from .build import build_all
        from .data_cache import build_cache
        from .evaluate import compute_embeddings, run_benchmarks
        text_embeddings()
        build_cache()
        build_all()
        compute_embeddings()
        run_benchmarks()
    from .report import write_report
    from .routing import run_routing
    write_report()
    run_routing()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
