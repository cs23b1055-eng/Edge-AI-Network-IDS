import numpy as np

def confusion(y, p, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm

def report(y, p, class_names):
    k = len(class_names); cm = confusion(y, p, k); tot = cm.sum()
    per = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = tot - tp - fn - fp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        per[name] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                     "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0,
                     "support": int(tp + fn)}
    present = [v for v in per.values() if v["support"] > 0]
    out = {"accuracy": float(np.trace(cm) / tot), "macro_precision": float(np.mean([v["precision"] for v in present])),
           "macro_recall": float(np.mean([v["recall"] for v in present])), "macro_f1": float(np.mean([v["f1"] for v in present])),
           "per_class": per, "confusion_matrix": cm.tolist(), "classes": class_names}
    if k == 2:  # attack = positive class 1
        out["false_positive_rate"] = per[class_names[1]]["fpr"]; out["false_negative_rate"] = per[class_names[1]]["fnr"]
        out["precision"], out["recall"], out["f1"] = (per[class_names[1]][m] for m in ("precision", "recall", "f1"))
    return out
