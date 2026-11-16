"""Phase 5: build train/val/test sets from the ORIGINAL CSVs (read-only).
 1. load + normalise labels, select FEATURES   2. drop exact duplicate (features,label) rows
 3. stratified 70/15/15 split on the 15 raw labels  4. fit Preprocessor on TRAIN ONLY, transform all
 5. write float32 binaries for the C++ trainer + raw split arrays (for experiments) + split report."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python_network"))
from preprocessing import FEATURES, MULTI_CLASSES, LABEL_TO_GROUP, Preprocessor, clean_label
DATA = os.path.abspath(os.path.join(ROOT, "..", "MachineLearningCVE"))
OUT = os.path.join(ROOT, "datasets", "processed"); os.makedirs(OUT, exist_ok=True)
SEED = 42

def write_bin(prefix, X, y):
    with open(prefix + "_X.bin", "wb") as f:
        np.array([X.shape[0], X.shape[1]], dtype=np.int32).tofile(f); np.ascontiguousarray(X, dtype=np.float32).tofile(f)
    with open(prefix + "_y.bin", "wb") as f:
        np.array([len(y)], dtype=np.int32).tofile(f); np.asarray(y, dtype=np.int32).tofile(f)

def main():
    Xs, labs, days = [], [], []
    for fi, f in enumerate(sorted(glob.glob(os.path.join(DATA, "*.csv")))):
        df = pd.read_csv(f, encoding="latin-1", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        # the label may carry a mojibake dash: normalise to ascii
        lab = df["Label"].astype(str).map(lambda s: clean_label(s.replace("\x96", "-").replace("�", "-").replace("ï¿½", "-").replace("–", "-")))
        Xs.append(df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
        labs.append(lab.to_numpy()); days.append(np.full(len(df), fi, dtype=np.int8)); del df
    X = np.concatenate(Xs); lab = np.concatenate(labs); day = np.concatenate(days); del Xs
    unknown = set(np.unique(lab)) - set(LABEL_TO_GROUP); assert not unknown, unknown
    n_raw = len(X)
    h = pd.util.hash_pandas_object(pd.DataFrame(np.column_stack([X, pd.factorize(lab)[0]])), index=False).to_numpy()
    keep = ~pd.Series(h).duplicated().to_numpy()
    X, lab, day = X[keep], lab[keep], day[keep]
    print(f"rows {n_raw} -> {len(X)} after dedup")
    # stratified split
    rng = np.random.default_rng(SEED); split = np.zeros(len(X), dtype=np.int8)  # 0 train 1 val 2 test
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c); rng.shuffle(idx); n = len(idx)
        nt, nv = int(round(.15 * n)), int(round(.15 * n))
        if n < 20: nt = nv = max(1, n // 5)   # tiny classes: still present in every split
        split[idx[:nt]] = 2; split[idx[nt:nt + nv]] = 1
    pre = Preprocessor().fit(X[split == 0])    # TRAIN ONLY
    binary = (lab != "BENIGN").astype(np.int32)
    grp = np.array([LABEL_TO_GROUP[l] for l in lab], dtype=object)
    multi = np.array([MULTI_CLASSES.index(g) if g else -1 for g in grp], dtype=np.int32)
    np.savez_compressed(os.path.join(OUT, "raw_split.npz"), X=X, lab=lab, day=day, split=split, binary=binary, multi=multi)
    rep = {"rows_raw": int(n_raw), "rows_after_dedup": int(len(X)), "seed": SEED, "splits": {}}
    for si, name in enumerate(["train", "val", "test"]):
        m = split == si; Z = pre.transform(X[m])
        write_bin(os.path.join(OUT, f"binary_{name}"), Z, binary[m])
        write_bin(os.path.join(OUT, f"multi_{name}"), Z[multi[m] >= 0], multi[m][multi[m] >= 0])
        rep["splits"][name] = {"rows": int(m.sum()), "labels": pd.Series(lab[m]).value_counts().to_dict()}
    pre.save(os.path.join(ROOT, "models", "preprocessing.json"),
             {"multi_classes": MULTI_CLASSES, "binary_classes": ["BENIGN", "ATTACK"], "fit_rows": int((split == 0).sum())})
    json.dump(rep, open(os.path.join(ROOT, "results", "split_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
main()
