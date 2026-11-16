"""Phase 6: baselines + experiments + final models. Everything reported here is measured by this script."""
import json, os, sys, time
import numpy as np
from common import *
from preprocessing import FEATURES, MULTI_CLASSES
from metrics import report
from engine import NativeModel
from sklearn.linear_model import LogisticRegression

RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models"); os.makedirs(MOD, exist_ok=True)
BIN = ["BENIGN", "ATTACK"]
Xtr, ytr = read_xy(f"{PROC}/binary_train"); Xva, yva = read_xy(f"{PROC}/binary_val"); Xte, yte = read_xy(f"{PROC}/binary_test")

def evaluate(model_path, X, y, names):
    m = NativeModel(model_path); t = time.perf_counter(); p = m.predict(X); dt = time.perf_counter() - t
    r = report(y, p, names); r["model_size_bytes"] = m.size_bytes; r["batch_inference_us_per_sample"] = dt / len(X) * 1e6; return r

results = {"baselines": {}, "experiments": [], "note": "model selection uses validation macro-F1; test metrics are reported, never used for selection"}
# ---- baselines
maj = np.zeros(len(yte), dtype=int); results["baselines"]["majority_class"] = report(yte, maj, BIN)
t = time.time(); lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(Xtr, ytr)
results["baselines"]["logistic_regression"] = report(yte, lr.predict(Xte), BIN); results["baselines"]["logistic_regression"]["train_seconds"] = time.time() - t
print("baselines done", flush=True)

# ---- feature-subset variants (columns sliced from the already-preprocessed matrices)
BASIC = [f for f in FEATURES if f not in ("Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward", "act_data_pkt_fwd")]
subsets = {"all57": FEATURES, "no_dest_port": [f for f in FEATURES if f != "Destination Port"],
           "no_port_no_tcp_state": BASIC}
def variant(name):
    d = os.path.join(PROC, "exp_" + name); os.makedirs(d, exist_ok=True)
    idx = [FEATURES.index(f) for f in subsets[name]]
    for s, (X, y) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items(): write_xy(f"{d}/binary_{s}", X[:, idx], y)
    return d, idx

def run(name, **kw):
    fs = kw.pop("features", "all57"); d, idx = variant(fs) if fs != "all57" else (PROC, list(range(len(FEATURES))))
    out = os.path.join(RES, f"exp_{name}.bin")
    log, secs = train("binary", 2, out, prefix_dir=d, epochs=12, patience=3, **kw)
    best = max(log[:-1], key=lambda e: e["val_macro_f1"]) if len(log) > 1 else {}
    r = evaluate(out, Xte[:, idx], yte, BIN)
    row = {"name": name, "features": fs, "n_features": len(idx), "params": kw, "epochs_run": len(log) - 1, "train_seconds": secs,
           "val_macro_f1": log[-1]["best_val_macro_f1"], "test_accuracy": r["accuracy"], "test_precision": r["precision"], "test_recall": r["recall"],
           "test_f1": r["f1"], "test_fpr": r["false_positive_rate"], "test_fnr": r["false_negative_rate"], "model_bytes": r["model_size_bytes"]}
    results["experiments"].append(row); os.remove(out)
    print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("val", "test_f", "test_acc"))}, flush=True)
    json.dump(results, open(f"{RES}/experiments.json", "w"), indent=1)

for h in ["32", "64,32", "128,64", "128,64,32"]: run(f"hidden_{h}", hidden=h)
for lr_ in [0.0003, 0.003, 0.01]: run(f"lr_{lr_}", lr=lr_)
for b in [64, 1024]: run(f"batch_{b}", batch=b)
run("opt_sgd_momentum_lr0.05", opt="sgd", lr=0.05)
run("cw_power_0", cw=0.0); run("cw_power_1", cw=1.0)
for fs in ["no_dest_port", "no_port_no_tcp_state"]: run(f"features_{fs}", features=fs)

# ---- generalisation to an UNSEEN attack family: train without PortScan+DDoS (Friday), test on them
S = np.load(f"{PROC}/raw_split.npz", allow_pickle=True); lab, split = S["lab"], S["split"]
held = np.isin(lab, ["PortScan", "DDoS"]); trm = (split == 0)
d = os.path.join(PROC, "exp_holdout"); os.makedirs(d, exist_ok=True)
write_xy(f"{d}/binary_train", Xtr[~held[trm]], ytr[~held[trm]]); write_xy(f"{d}/binary_val", Xva[~held[split == 1]], yva[~held[split == 1]])
out = os.path.join(RES, "exp_holdout.bin"); log, _ = train("binary", 2, out, prefix_dir=d, tag="binary", epochs=12, patience=3)
m = NativeModel(out); tl = lab[split == 2]; Xh = Xte[np.isin(tl, ["PortScan", "DDoS"])]
fam = {f: float((m.predict(Xte[tl == f]) == 1).mean()) for f in ["PortScan", "DDoS", "DoS Hulk", "FTP-Patator"]}
results["unseen_family_generalisation"] = {"trained_without": ["PortScan", "DDoS"], "attack_recall_on_family_in_test": fam,
    "benign_fpr": float((m.predict(Xte[yte == 0]) == 1).mean()), "note": "PortScan/DDoS recall is for families never seen in training; others were seen"}
os.remove(out); print("holdout", results["unseen_family_generalisation"], flush=True)
json.dump(results, open(f"{RES}/experiments.json", "w"), indent=1)
