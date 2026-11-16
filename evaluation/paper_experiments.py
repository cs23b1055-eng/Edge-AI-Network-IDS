"""Controlled experiments for the paper (all measured here, multi-seed, same split/budget).
 A. cumulative ablation (add one component at a time)  B. feature ablation  C. leave-one-attack-family-out generalisation."""
import json, os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import *
from preprocessing import FEATURES, LABEL_TO_GROUP, signed_log1p
from metrics import report
from engine import NativeModel

RES = os.path.join(ROOT, "results"); TMP = os.path.join(RES, "tmp_models"); os.makedirs(TMP, exist_ok=True)
SEEDS = [1, 2, 3, 4, 5]; BIN = ["BENIGN", "ATTACK"]; EPOCHS = 10; PAT = 3
S = np.load(f"{PROC}/raw_split.npz", allow_pickle=True); lab, split = S["lab"], S["split"]
Xtr, ytr = read_xy(f"{PROC}/binary_train"); Xva, yva = read_xy(f"{PROC}/binary_val"); Xte, yte = read_xy(f"{PROC}/binary_test")

# --- z-score-only variant (no signed-log1p) for the preprocessing ablation; fitted on TRAIN only
raw = S["X"].astype(np.float64); raw[~np.isfinite(raw)] = np.nan
med = np.nan_to_num(np.nanmedian(raw[split == 0], axis=0)); raw = np.where(np.isnan(raw), med, raw)
mu, sd = raw[split == 0].mean(0), raw[split == 0].std(0); sd[sd < 1e-9] = 1
Z = np.clip((raw - mu) / sd, -10, 10).astype(np.float32); del raw
dz = os.path.join(PROC, "exp_zonly"); os.makedirs(dz, exist_ok=True)
for i, s in enumerate(["train", "val", "test"]): write_xy(f"{dz}/binary_{s}", Z[split == i], S["binary"][split == i])
Zte = Z[split == 2]; del Z

def variant_dir(name, cols):
    d = os.path.join(PROC, "exp_" + name); os.makedirs(d, exist_ok=True)
    for s, (X, y) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items(): write_xy(f"{d}/binary_{s}", X[:, cols], y)
    return d
ALL = list(range(len(FEATURES))); NOPORT = [i for i, f in enumerate(FEATURES) if f != "Destination Port"]
NOTCP = [i for i, f in enumerate(FEATURES) if f not in ("Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward", "act_data_pkt_fwd")]
dirs = {"all57": (PROC, ALL), "no_port": (variant_dir("no_port", NOPORT), NOPORT), "no_port_no_tcp_state": (variant_dir("no_tcp", NOTCP), NOTCP)}

def job(tag, d, test_X, hidden, cw, seed, test_y=None):
    out = os.path.join(TMP, f"{tag}_{seed}.bin")
    log, secs = train("binary", 2, out, prefix_dir=d, hidden=hidden, cw=cw, seed=seed, epochs=EPOCHS, patience=PAT)
    m = NativeModel(out); r = report(test_y if test_y is not None else yte, m.predict(test_X), BIN); os.remove(out)
    return {k: r[k] for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")} | {"epochs": len(log) - 1}

def run_many(spec):
    with ThreadPoolExecutor(4) as ex: return list(ex.map(lambda a: job(*a), spec))
def summarize(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1)), "runs": [r[k] for r in rows]} for k in rows[0] if k != "epochs"}

out = {"seeds": SEEDS, "max_epochs": EPOCHS, "note": "mean/std over seeds; same split, same epoch budget, binary task, test split"}
# A. cumulative ablation
cfgs = [("A0_linear_zscore", dz, Zte, "", 0.0), ("A1_+log1p", PROC, Xte, "", 0.0), ("A2_+hidden64", PROC, Xte, "64", 0.0),
        ("A3_+hidden32(64-32)", PROC, Xte, "64,32", 0.0), ("A4_+class_weights(final)", PROC, Xte, "64,32", 0.5)]
out["ablation"] = {}
for name, d, tx, h, cw in cfgs:
    t = time.time(); out["ablation"][name] = summarize(run_many([(name, d, tx, h, cw, s) for s in SEEDS])); print(name, round(out["ablation"][name]["f1"]["mean"], 5), f"{time.time()-t:.0f}s", flush=True)
    json.dump(out, open(f"{RES}/paper_experiments.json", "w"), indent=1)
# B. feature ablation (final architecture)
out["features"] = {}
for name, (d, cols) in dirs.items():
    out["features"][name] = summarize(run_many([("F_" + name, d, Xte[:, cols], "64,32", 0.5, s) for s in SEEDS])); out["features"][name]["n_features"] = len(cols); print(name, round(out["features"][name]["f1"]["mean"], 5), flush=True)
    json.dump(out, open(f"{RES}/paper_experiments.json", "w"), indent=1)
# C. leave-one-family-out
grp = np.array([LABEL_TO_GROUP[l] or "Rare" for l in lab], dtype=object); out["leave_one_family_out"] = {}
for fam in ["DoS", "DDoS", "PortScan", "Brute Force", "Web Attack", "Bot"]:
    d = os.path.join(PROC, "exp_lofo"); os.makedirs(d, exist_ok=True)
    for i, s in ((0, "train"), (1, "val")): write_xy(f"{d}/binary_{s}", (Xtr if i == 0 else Xva)[grp[split == i] != fam], (ytr if i == 0 else yva)[grp[split == i] != fam])
    write_xy(f"{d}/binary_test", Xte, yte)
    held = grp[split == 2] == fam; Xh = Xte[held]; Xb = Xte[yte == 0]; rows = []
    def lofo(seed):
        out_p = os.path.join(TMP, f"lofo_{fam.replace(' ','_')}_{seed}.bin")
        train("binary", 2, out_p, prefix_dir=d, seed=seed, epochs=EPOCHS, patience=PAT); m = NativeModel(out_p)
        r = {"recall_on_unseen_family": float((m.predict(Xh) == 1).mean()), "benign_fpr": float((m.predict(Xb) == 1).mean())}; os.remove(out_p); return r
    with ThreadPoolExecutor(3) as ex: rows = list(ex.map(lofo, SEEDS[:3]))
    out["leave_one_family_out"][fam] = {"n_test_flows": int(held.sum()), **summarize(rows)}; print("LOFO", fam, round(out["leave_one_family_out"][fam]["recall_on_unseen_family"]["mean"], 4), flush=True)
    json.dump(out, open(f"{RES}/paper_experiments.json", "w"), indent=1)
print("DONE")
