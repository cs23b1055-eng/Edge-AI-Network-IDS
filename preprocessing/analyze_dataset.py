"""Phase 2: memory-light dataset analysis. Read-only on the original CSVs.
Writes results/dataset_analysis.json (consumed by documentation/DATASET_REPORT.md)."""
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.abspath(os.path.join(ROOT, "..", "MachineLearningCVE"))

def main():
    rep = {"dataset_dir": DATA, "files": {}}
    hashes, labels_all, file_ids = [], [], []
    cols0 = None
    inf_by_col, nan_by_col, neg_by_col = {}, {}, {}
    const_tracker = {}
    for fi, f in enumerate(sorted(glob.glob(os.path.join(DATA, "*.csv")))):
        df = pd.read_csv(f, encoding="latin-1", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        cols = list(df.columns)
        if cols0 is None: cols0 = cols
        lab = df["Label"].astype(str).str.strip().str.replace("\x96", "-", regex=False).str.replace("�", "-", regex=False)
        X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").astype(np.float64)
        a = X.to_numpy()
        isinf, isnan = np.isinf(a), np.isnan(a)
        for i, c in enumerate(X.columns):
            if isinf[:, i].any(): inf_by_col[c] = inf_by_col.get(c, 0) + int(isinf[:, i].sum())
            if isnan[:, i].any(): nan_by_col[c] = nan_by_col.get(c, 0) + int(isnan[:, i].sum())
            if (a[:, i] < 0).any(): neg_by_col[c] = neg_by_col.get(c, 0) + int((a[:, i] < 0).sum())
            u = const_tracker.setdefault(c, set())
            if len(u) < 3: u.update(np.unique(a[~isnan[:, i] & ~isinf[:, i], i])[:3].tolist())
        h = pd.util.hash_pandas_object(pd.concat([X, lab.rename("Label")], axis=1), index=False).to_numpy()
        hashes.append(h); labels_all.append(lab.to_numpy()); file_ids.append(np.full(len(df), fi))
        rep["files"][os.path.basename(f)] = {
            "rows": len(df), "n_columns": len(cols), "same_columns_as_first": cols == cols0,
            "nan": int(isnan.sum()), "inf": int(isinf.sum()),
            "exact_duplicate_rows_within_file": int(pd.Series(h).duplicated().sum()),
            "labels": lab.value_counts().to_dict()}
        print(os.path.basename(f), len(df), flush=True)
        del df, X, a, isinf, isnan
    h = np.concatenate(hashes); lab = np.concatenate(labels_all); fid = np.concatenate(file_ids)
    dup = pd.Series(h).duplicated().to_numpy()
    rep.update({
        "columns": cols0, "n_features": len(cols0) - 1, "label_column": "Label", "total_rows": int(len(h)),
        "class_distribution": pd.Series(lab).value_counts().to_dict(),
        "total_duplicate_rows": int(dup.sum()), "rows_after_dedup": int((~dup).sum()),
        "class_distribution_after_dedup": pd.Series(lab[~dup]).value_counts().to_dict(),
        "nan_by_col": nan_by_col, "inf_by_col": inf_by_col, "nan_total": sum(nan_by_col.values()),
        "inf_total": sum(inf_by_col.values()), "negative_value_cols": neg_by_col,
        "constant_columns": [c for c, u in const_tracker.items() if len(u) <= 1],
        "label_by_file": {k: v for k, v in pd.crosstab(pd.Series(fid, name="f"), pd.Series(lab, name="l")).to_dict().items()},
        "conflicting_labels_same_features": "not computed (hash includes label)",
        "notes": {"has_timestamp": "Timestamp" in cols0, "has_ip_columns": any("IP" in c for c in cols0),
                  "duplicate_header": [c for c in cols0 if c.endswith(".1")]}})
    json.dump(rep, open(os.path.join(ROOT, "results", "dataset_analysis.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("label_by_file", "columns", "files")}, indent=1, default=str))
main()
