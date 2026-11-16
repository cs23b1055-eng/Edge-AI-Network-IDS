"""Single source of truth for feature definitions and the train/inference transform.

The SAME Preprocessor class is used offline (fit on the training split only) and online
(loaded from JSON), which guarantees training/inference consistency.
"""
import json
import numpy as np

# 57 CIC-style flow features. Names are the dataset's own column names (stripped). Chosen because
# each can be derived from a packet stream. Excluded (documented in DATASET_REPORT.md): 8 constant
# columns, the duplicated 'Fwd Header Length.1', Subflow* (identical to totals), Active*/Idle*
# (need CICFlowMeter's activity-timeout state machine), Bwd PSH/URG (constant 0 in the data).
FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
]

# Multiclass categories. Infiltration (36 rows) and Heartbleed (11 rows) are too rare to learn or
# evaluate; they are used only as 'malicious' in the binary task.
MULTI_CLASSES = ["BENIGN", "DoS", "DDoS", "PortScan", "Brute Force", "Web Attack", "Bot"]
LABEL_TO_GROUP = {
    "BENIGN": "BENIGN", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS", "DDoS": "DDoS", "PortScan": "PortScan", "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force", "Web Attack - Brute Force": "Web Attack", "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack", "Bot": "Bot",
    "Infiltration": None, "Heartbleed": None,
}

def clean_label(s: str) -> str:
    """Normalise raw labels: the CSVs contain a mis-encoded en-dash in 'Web Attack - ...'."""
    for bad in ("\x96", "\ufffd", "\u00ef\u00bf\u00bd", "\u2013"):
        s = s.replace(bad, "-")
    return " ".join(s.split())

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

class Preprocessor:
    """NaN/inf -> NaN -> train median; signed log1p; z-score (train mean/std); clip to +-10."""
    CLIP = 10.0

    def __init__(self, features=None):
        self.features = list(features or FEATURES)
        self.median = self.mean = self.std = None

    def _clean(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X[None, :]
        if X.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features, got {X.shape[1]}")
        X = X.copy()
        X[~np.isfinite(X)] = np.nan
        return X

    def fit(self, X):
        X = self._clean(X)
        self.median = np.nan_to_num(np.nanmedian(X, axis=0))
        Z = signed_log1p(np.where(np.isnan(X), self.median, X))
        self.mean = Z.mean(axis=0)
        s = Z.std(axis=0)
        self.std = np.where(s < 1e-9, 1.0, s)
        return self

    def transform(self, X):
        if self.median is None: raise RuntimeError("Preprocessor not fitted")
        X = self._clean(X)
        X = np.where(np.isnan(X), self.median, X)
        Z = (signed_log1p(X) - self.mean) / self.std
        return np.clip(Z, -self.CLIP, self.CLIP).astype(np.float32)

    def select(self, features):
        """Sub-preprocessor over a feature subset (for feature-combination experiments)."""
        idx = [self.features.index(f) for f in features]
        p = Preprocessor(features)
        p.median, p.mean, p.std = self.median[idx], self.mean[idx], self.std[idx]
        return p

    def to_dict(self, extra=None):
        d = {"features": self.features, "median": self.median.tolist(), "mean": self.mean.tolist(),
             "std": self.std.tolist(), "clip": self.CLIP, "transform": "nan/inf->median; signed_log1p; zscore; clip"}
        d.update(extra or {})
        return d

    def save(self, path, extra=None):
        json.dump(self.to_dict(extra), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); p = cls(d["features"])
        p.median, p.mean, p.std = (np.array(d[k]) for k in ("median", "mean", "std"))
        return p
