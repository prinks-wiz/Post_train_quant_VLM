"""The four live configurations (stdlib only, so the UI server never touches numpy/onnxruntime)."""

CONFIGS = {
    "cascade": {
        "spec": "int8_dyn_d12_aligned_r128-224_f20__np",
        "label": "INT8-dyn cascade 128 → 224",
        "note": "zero-shot · aligned heads · one shared weight copy",
    },
    "int8_224": {
        "spec": "int8_dyn_d12_aligned_r224__np",
        "label": "INT8-dyn 224 px",
        "note": "zero-shot · single pass · tightest tail",
    },
    "int4_224": {
        "spec": "int4_a4_d12_aligned_r224__np",
        "label": "INT4 (a4) 224 px",
        "note": "zero-shot · lowest RAM · INT8 compute (accuracy_level=4)",
    },
    "probe": {
        "spec": "int8_dyn_d12_probe_r128-224_f10__np",
        "label": "Probe cascade 128 → 224",
        "note": "uses 5,000 train labels, not zero-shot",
    },
}
DEFAULT = "cascade"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
