"""Train + evaluate + export the deployed models (binary and multiclass) with the chosen configuration."""
import argparse, json, os, shutil, sys, time
import numpy as np
from common import *
from preprocessing import FEATURES, MULTI_CLASSES, Preprocessor
from metrics import report
from engine import NativeModel
ap = argparse.ArgumentParser()
ap.add_argument("--hidden", default="64,32"); ap.add_argument("--lr", type=float, default=0.001); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--features", choices=["all57", "no_dest_port"], default="all57")
a = ap.parse_args()
feats = FEATURES if a.features == "all57" else [f for f in FEATURES if f != "Destination Port"]
idx = [FEATURES.index(f) for f in feats]; MOD = os.path.join(ROOT, "models"); RES = os.path.join(ROOT, "results")
d = os.path.join(PROC, "final_" + a.features); os.makedirs(d, exist_ok=True)
data = {}
for task in ("binary", "multi"):
    for s in ("train", "val", "test"):
        X, y = read_xy(f"{PROC}/{task}_{s}"); write_xy(f"{d}/{task}_{s}", X[:, idx], y); data[(task, s)] = (X[:, idx], y)
cfg = {"hidden": a.hidden, "lr": a.lr, "batch": a.batch, "max_epochs": a.epochs, "optimizer": "adam", "class_weight_power": 0.5, "seed": 42, "features": a.features, "n_features": len(feats)}
out = {"config": cfg}
for task, k, names, fn in (("binary", 2, ["BENIGN", "ATTACK"], "binary_model.bin"), ("multi", len(MULTI_CLASSES), MULTI_CLASSES, "multiclass_model.bin")):
    log, secs = train(task, k, os.path.join(MOD, fn), prefix_dir=d, hidden=a.hidden, lr=a.lr, batch=a.batch, epochs=a.epochs, patience=6)
    m = NativeModel(os.path.join(MOD, fn)); X, y = data[(task, "test")]
    r = report(y, m.predict(X), names); r["train_seconds"] = secs; r["epochs_run"] = len(log) - 1; r["model_size_bytes"] = m.size_bytes; r["training_log"] = log[:-1]
    out["binary" if task == "binary" else "multiclass"] = r
    print(task, {k_: round(v, 5) for k_, v in r.items() if isinstance(v, float)}, flush=True)
# preprocessing config restricted to the model's feature list (fit on the full 57, train split only)
full = os.path.join(PROC, "preprocessing_full.json")   # pristine 57-feature fit (train split only)
if not os.path.exists(full): shutil.copy(os.path.join(MOD, "preprocessing.json"), full)
pre = Preprocessor.load(full); assert len(pre.features) == len(FEATURES)
pre.select(feats).save(os.path.join(MOD, "preprocessing.json"), {"multi_classes": MULTI_CLASSES, "binary_classes": ["BENIGN", "ATTACK"], "training_config": cfg})
json.dump(out, open(os.path.join(RES, "final_evaluation.json"), "w"), indent=1)
