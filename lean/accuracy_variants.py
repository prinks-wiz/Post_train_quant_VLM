"""Full-test accuracy for derived deployment specs (e.g. OpenCV-free preprocessing `__np`), computed
with the runtime's own preprocessing/heads/cascade rule (batched). Appends to results/lean/accuracy.csv.
Mine."""
import json

import numpy as np
import pandas as pd

from .runtime import LEAN_ART, ROOT, Head, margin, preprocess, preprocess_np, session

ACC = ROOT / "results" / "lean" / "accuracy.csv"


def spec_accuracy(spec, test, y, batch=200):
    prep = preprocess_np if spec.get("prep") == "numpy" else preprocess
    sess = session(LEAN_ART / spec["onnx"], lean_mem=False)
    lg = {}
    for r in spec["stages"]:
        pos = np.load(LEAN_ART / "pos" / f"pos_r{r}.npy")
        F = np.concatenate([sess.run(None, {"pixel_values": prep(test[i:i + batch], r), "pos": pos})[0]
                            for i in range(0, len(test), batch)])
        lg[r] = Head(LEAN_ART / spec["heads"][str(r)]).logits(F)
    r1 = spec["stages"][0]
    pred, esc = lg[r1].argmax(1), np.zeros(len(y), bool)
    if len(spec["stages"]) > 1:
        esc = margin(lg[r1]) < spec["threshold"]
        pred = np.where(esc, lg[spec["stages"][1]].argmax(1), pred)
    return float((pred == y).mean()), float(esc.mean())


def main():
    test, y = np.load(LEAN_ART / "test_raw.npy"), np.load(LEAN_ART / "test_labels.npy")
    acc = pd.read_csv(ACC)
    names = sorted(p.stem for p in (LEAN_ART / "specs").glob("*__np*.json"))
    rows, cache = [], {}
    for n in names:
        spec = json.loads((LEAN_ART / "specs" / f"{n}.json").read_text())
        key = n.replace("__np_nopack", "__np")   # prepacking doesn't change numerics
        if key not in cache:
            cache[key] = spec_accuracy(spec, test, y)
            print(n, cache[key], flush=True)
        a, f = cache[key]
        rows.append({"name": n, "onnx": spec["onnx"], "stages": spec["stages"], "precision": spec["precision"],
                     "head_kind": spec["head_kind"], "threshold": spec.get("threshold"), "acc": a,
                     "frac_second": f, "model_file_mb": (LEAN_ART / spec["onnx"]).stat().st_size / 2**20})
    acc = pd.concat([acc[~acc.name.isin(names)], pd.DataFrame(rows)], ignore_index=True)
    acc.to_csv(ACC, index=False)


if __name__ == "__main__":
    main()
