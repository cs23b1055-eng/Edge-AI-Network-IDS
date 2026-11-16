"""Figures + error analysis for the paper. Stage 1 needs only the final models; stage 2 (--stage2) needs paper_experiments.json."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
from preprocessing import FEATURES, LABEL_TO_GROUP, MULTI_CLASSES
from engine import NativeModel
FIG = os.path.join(ROOT, "documentation", "figures"); RES = os.path.join(ROOT, "results")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
ev = json.load(open(f"{RES}/final_evaluation.json"))

def arch():
    fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    boxes = [("Packets\n(live / PCAP)", "Scapy"), ("Flow assembly", "5-tuple, FIN/RST,\nidle timeout"), ("57 flow\nfeatures", "durations, IAT,\nlengths, flags"),
             ("Preprocess", "median, log1p,\nz-score (train fit)"), ("C++ MLP\n(ctypes)", "57-64-32-2 / 7"), ("Alert +\nDashboard", "Streamlit")]
    for i, (t, s) in enumerate(boxes):
        x = 1 + i * 16.5; ax.add_patch(FancyBboxPatch((x, 9), 14, 14, boxstyle="round,pad=0.3", fc="#e8f0fa" if i != 4 else "#fde9d4", ec="#3b5b8c"))
        ax.text(x + 7, 18, t, ha="center", va="center", fontweight="bold", fontsize=8); ax.text(x + 7, 5.5, s, ha="center", va="center", fontsize=7, color="#444")
        if i < 5: ax.annotate("", xy=(x + 16.2, 16), xytext=(x + 14.4, 16), arrowprops=dict(arrowstyle="->", color="#3b5b8c"))
    fig.savefig(f"{FIG}/fig_pipeline.png", bbox_inches="tight"); plt.close(fig)

def mlp_diagram():
    fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off"); sizes = [57, 64, 32, 2]; names = ["Input\n(57 features)", "Hidden 1\n64 + ReLU", "Hidden 2\n32 + ReLU", "Output\nsoftmax (2 or 7)"]
    pos = []
    for i, n in enumerate(sizes):
        k = min(n, 8); ys = np.linspace(0.15, 0.85, k) if k > 1 else [0.5]; pos.append([(0.1 + i * 0.27, y) for y in ys])
    for i in range(3):
        for a in pos[i]:
            for b in pos[i + 1]: ax.plot([a[0], b[0]], [a[1], b[1]], color="#9db4d4", lw=0.3, zorder=1)
    for i, col in enumerate(pos):
        for (x, y) in col: ax.scatter(x, y, s=60, color=["#3b5b8c", "#5b8c5b", "#5b8c5b", "#c9763b"][i], zorder=2)
        ax.text(0.1 + i * 0.27, -0.03, names[i], ha="center", va="top", fontsize=8)
    ax.text(0.1, 0.93, "(8 of 57 drawn)", fontsize=6, color="#666"); ax.set_xlim(0, 0.95); ax.set_ylim(-0.25, 1.0)
    fig.savefig(f"{FIG}/fig_mlp.png", bbox_inches="tight"); plt.close(fig)

def confusions():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axs[0], "binary", "Binary (row-normalised)"), (axs[1], "multiclass", "7-class (row-normalised)")):
        cm = np.array(ev[key]["confusion_matrix"], float); n = cm / cm.sum(1, keepdims=True); names = ev[key]["classes"]
        ax.imshow(n, cmap="Blues", vmin=0, vmax=1); ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_yticklabels(names); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)): ax.text(j, i, f"{cm[i, j]:,.0f}\n{n[i, j]*100:.1f}%" if len(names) == 2 else f"{n[i, j]*100:.1f}", ha="center", va="center", fontsize=8 if len(names) == 2 else 6.5, color="white" if n[i, j] > .5 else "black")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_confusion.png", bbox_inches="tight"); plt.close(fig)

def curves():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, k in zip(axs, ("binary", "multiclass")):
        lg = ev[k]["training_log"]; e = [x["epoch"] for x in lg]
        ax.plot(e, [x["train_loss"] for x in lg], label="train loss (weighted)"); ax.plot(e, [x["val_loss"] for x in lg], label="val loss"); ax2 = ax.twinx()
        ax2.plot(e, [x["val_macro_f1"] for x in lg], color="green", ls="--", label="val macro-F1"); ax.set_xlabel("epoch"); ax.set_title(k); ax2.set_ylim(0.5, 1.0)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_training.png", bbox_inches="tight"); plt.close(fig)

def error_analysis():
    d = np.load(f"{PROC}/raw_split.npz", allow_pickle=True); m = d["split"] == 2; lab = d["lab"][m]; Xraw = d["X"][m]
    Xte, yte = read_xy(f"{PROC}/binary_test"); mb = NativeModel(os.path.join(ROOT, "models", "binary_model.bin")); mm = NativeModel(os.path.join(ROOT, "models", "multiclass_model.bin"))
    pb = mb.predict_proba(Xte)[:, 1]; pred = pb >= 0.5; res = {"false_negatives_by_label": {}, "false_positives": int(((~(lab != "BENIGN")) & pred).sum())}
    for l in sorted(set(lab) - {"BENIGN"}):
        s = lab == l; res["false_negatives_by_label"][l] = {"n": int(s.sum()), "missed": int((~pred[s]).sum()), "miss_rate": float((~pred[s]).mean())}
    atk = lab != "BENIGN"; res["conf_of_errors"] = {"fn_mean_p_attack": float(pb[atk & ~pred].mean()), "fp_mean_p_attack": float(pb[~atk & pred].mean()),
        "fn_p_attack_below_0.1": float((pb[atk & ~pred] < 0.1).mean())}
    # multiclass top confusions
    cm = np.array(ev["multiclass"]["confusion_matrix"]); names = ev["multiclass"]["classes"]; pairs = [(int(cm[i, j]), names[i], names[j]) for i in range(7) for j in range(7) if i != j]
    res["top_multiclass_confusions"] = [{"count": c, "true": a, "pred": b} for c, a, b in sorted(pairs, reverse=True)[:6]]
    # feature profile of missed flows vs class medians (raw feature units)
    keys = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Flow Packets/s", "SYN Flag Count", "PSH Flag Count", "Init_Win_bytes_forward"]
    ix = [FEATURES.index(k) for k in keys]; prof = {}
    for l in [l for l, v in res["false_negatives_by_label"].items() if v["missed"] >= 5]:
        s = (lab == l); prof[l] = {"missed_median": dict(zip(keys, np.nanmedian(Xraw[s & ~pred][:, ix], 0).tolist())), "detected_median": dict(zip(keys, np.nanmedian(Xraw[s & pred][:, ix], 0).tolist())) if (s & pred).any() else None}
    prof["BENIGN_median"] = dict(zip(keys, np.nanmedian(Xraw[lab == "BENIGN"][:, ix], 0).tolist())); res["feature_profiles"] = prof
    json.dump(res, open(f"{RES}/error_analysis.json", "w"), indent=1)
    # figure: miss rate per attack label + confidence of errors
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); ls = sorted(res["false_negatives_by_label"], key=lambda k: -res["false_negatives_by_label"][k]["miss_rate"])
    axs[0].barh(ls, [res["false_negatives_by_label"][k]["miss_rate"] * 100 for k in ls], color="#c9763b"); axs[0].invert_yaxis(); axs[0].set_xlabel("missed (% of that attack in test)")
    for i, k in enumerate(ls): axs[0].text(res["false_negatives_by_label"][k]["miss_rate"] * 100 + .1, i, f"{res['false_negatives_by_label'][k]['missed']}/{res['false_negatives_by_label'][k]['n']}", va="center", fontsize=7)
    axs[0].set_title("Binary model: false negatives by attack type")
    axs[1].hist(pb[atk & ~pred], bins=20, range=(0, .5), alpha=.8, label=f"false negatives (n={int((atk&~pred).sum())})", color="#c9763b"); axs[1].hist(pb[~atk & pred], bins=20, range=(.5, 1), alpha=.8, label=f"false positives (n={int((~atk&pred).sum())})", color="#3b5b8c")
    axs[1].set_xlabel("P(attack) assigned by the model"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7); axs[1].set_title("How confident are the errors?")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_errors.png", bbox_inches="tight"); plt.close(fig); return res

def pca_plot():
    d = np.load(f"{PROC}/raw_split.npz", allow_pickle=True); m = d["split"] == 2; lab = d["lab"][m]; X, _ = read_xy(f"{PROC}/binary_test")
    rng = np.random.default_rng(0); grp = np.array([LABEL_TO_GROUP[l] or "Rare" for l in lab], dtype=object)
    idx = np.concatenate([rng.choice(np.flatnonzero(grp == g), min(2500, (grp == g).sum()), replace=False) for g in np.unique(grp)])
    Xs = X[idx] - X[idx].mean(0); U, S, Vt = np.linalg.svd(Xs, full_matrices=False); P = Xs @ Vt[:2].T; var = (S ** 2 / (S ** 2).sum())[:2]
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); cols = dict(zip(["BENIGN", "DoS", "DDoS", "PortScan", "Brute Force", "Web Attack", "Bot", "Rare"], plt.cm.tab10.colors))
    for g in ["BENIGN", "DoS", "DDoS", "PortScan", "Brute Force", "Web Attack", "Bot"]:
        s = grp[idx] == g; ax.scatter(P[s, 0], P[s, 1], s=3, alpha=.45, color=cols[g], label=g)
    ax.legend(markerscale=4, fontsize=7); ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var.)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var.)"); ax.set_title("PCA of standardised test features (2,500 flows per class)")
    fig.savefig(f"{FIG}/fig_pca.png", bbox_inches="tight"); plt.close(fig)

def stage2():
    pe = json.load(open(f"{RES}/paper_experiments.json")); fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    ab = pe["ablation"]; names = list(ab); x = np.arange(len(names))
    axs[0].bar(x - .2, [ab[n]["f1"]["mean"] for n in names], .4, yerr=[ab[n]["f1"]["std"] for n in names], capsize=3, label="F1 (attack)", color="#3b5b8c")
    axs[0].set_xticks(x); axs[0].set_xticklabels([n.split("_", 1)[1] for n in names], rotation=25, ha="right", fontsize=7); axs[0].set_ylim(min(ab[n]["f1"]["mean"] for n in names) - .02, 1.0); axs[0].set_title("Cumulative ablation (mean ± std, 5 seeds)"); axs[0].set_ylabel("test F1")
    lo = pe.get("leave_one_family_out", {}); fams = list(lo)
    if fams:
        axs[1].bar(fams, [lo[f]["recall_on_unseen_family"]["mean"] * 100 for f in fams], yerr=[lo[f]["recall_on_unseen_family"]["std"] * 100 for f in fams], capsize=3, color="#c9763b")
        axs[1].set_ylabel("recall on the withheld family (%)"); axs[1].set_title("Leave-one-attack-family-out (3 seeds)"); axs[1].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_ablation_lofo.png", bbox_inches="tight"); plt.close(fig)

if "--stage2" in sys.argv: stage2()
else:
    arch(); mlp_diagram(); confusions(); curves(); pca_plot(); r = error_analysis(); print(json.dumps({k: r[k] for k in ("false_negatives_by_label", "false_positives", "conf_of_errors", "top_multiclass_confusions")}, indent=1))
