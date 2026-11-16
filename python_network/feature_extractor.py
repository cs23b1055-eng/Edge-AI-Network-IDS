"""Flow -> 57-element feature vector in preprocessing.FEATURES order, following the CICFlowMeter-style
definitions used by the training dataset (times in microseconds, lengths = transport payload bytes)."""
import numpy as np
from flow_manager import Flow
from packet_capture import FIN, SYN, RST, PSH, ACK, URG, ECE, CWR
from preprocessing import FEATURES

def _stats(a):
    """max, min, mean, std(sample, ddof=1) of a list; zeros if empty."""
    if len(a) == 0: return 0.0, 0.0, 0.0, 0.0
    x = np.asarray(a, dtype=np.float64)
    return float(x.max()), float(x.min()), float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

def _iat(ts):
    """Inter-arrival times in microseconds from a sorted list of timestamps."""
    if len(ts) < 2: return []
    return list(np.diff(np.asarray(ts, dtype=np.float64)) * 1e6)

def _flag_count(pkts, bit): return sum(1 for p in pkts if p.flags & bit)

def extract(f: Flow) -> np.ndarray:
    fw, bw = f.fwd, f.bwd; allp = fw + bw
    dur_us = max(0.0, (f.last - f.start) * 1e6)
    fl = [p.payload for p in fw]; bl = [p.payload for p in bw]; al = fl + bl
    tf, tb = sum(fl), sum(bl)
    fmax, fmin, fmean, fstd = _stats(fl); bmax, bmin, bmean, bstd = _stats(bl)
    amax, amin, amean, astd = _stats(al)
    ts_all = sorted(t for t, _ in f.order); ts_f = sorted(p.ts for p in fw); ts_b = sorted(p.ts for p in bw)
    ia = _iat(ts_all); ifw = _iat(ts_f); ibw = _iat(ts_b)
    def iat4(x):
        if not x: return 0.0, 0.0, 0.0, 0.0
        mx, mn, mean, std = _stats(x); return mean, std, mx, mn
    dur_s = dur_us / 1e6
    # CICFlowMeter reports Infinity for zero-duration rates; the Preprocessor maps inf/NaN -> train median.
    bps = (tf + tb) / dur_s if dur_s > 0 else float("inf")
    pps = len(allp) / dur_s if dur_s > 0 else float("inf")
    tcp_f = [p for p in fw if p.proto == 6]; tcp_b = [p for p in bw if p.proto == 6]
    allflags = lambda bit: _flag_count(allp, bit)
    v = {
        "Destination Port": f.dport, "Flow Duration": dur_us, "Total Fwd Packets": len(fw), "Total Backward Packets": len(bw),
        "Total Length of Fwd Packets": tf, "Total Length of Bwd Packets": tb,
        "Fwd Packet Length Max": fmax, "Fwd Packet Length Min": fmin, "Fwd Packet Length Mean": fmean, "Fwd Packet Length Std": fstd,
        "Bwd Packet Length Max": bmax, "Bwd Packet Length Min": bmin, "Bwd Packet Length Mean": bmean, "Bwd Packet Length Std": bstd,
        "Flow Bytes/s": bps, "Flow Packets/s": pps,
        "Flow IAT Mean": iat4(ia)[0], "Flow IAT Std": iat4(ia)[1], "Flow IAT Max": iat4(ia)[2], "Flow IAT Min": iat4(ia)[3],
        "Fwd IAT Total": sum(ifw), "Fwd IAT Mean": iat4(ifw)[0], "Fwd IAT Std": iat4(ifw)[1], "Fwd IAT Max": iat4(ifw)[2], "Fwd IAT Min": iat4(ifw)[3],
        "Bwd IAT Total": sum(ibw), "Bwd IAT Mean": iat4(ibw)[0], "Bwd IAT Std": iat4(ibw)[1], "Bwd IAT Max": iat4(ibw)[2], "Bwd IAT Min": iat4(ibw)[3],
        "Fwd PSH Flags": _flag_count(fw, PSH), "Fwd URG Flags": _flag_count(fw, URG),
        "Fwd Header Length": sum(p.hdr_len for p in fw), "Bwd Header Length": sum(p.hdr_len for p in bw),
        "Fwd Packets/s": len(fw) / dur_s if dur_s > 0 else float("inf"), "Bwd Packets/s": len(bw) / dur_s if dur_s > 0 else float("inf"),
        "Min Packet Length": amin, "Max Packet Length": amax, "Packet Length Mean": amean, "Packet Length Std": astd, "Packet Length Variance": astd ** 2,
        "FIN Flag Count": allflags(FIN), "SYN Flag Count": allflags(SYN), "RST Flag Count": allflags(RST), "PSH Flag Count": allflags(PSH),
        "ACK Flag Count": allflags(ACK), "URG Flag Count": allflags(URG), "CWE Flag Count": allflags(CWR), "ECE Flag Count": allflags(ECE),
        "Down/Up Ratio": (len(bw) // len(fw)) if len(fw) else 0,
        "Average Packet Size": (sum(al) / len(al)) if al else 0.0, "Avg Fwd Segment Size": fmean, "Avg Bwd Segment Size": bmean,
        "Init_Win_bytes_forward": tcp_f[0].window if tcp_f else -1, "Init_Win_bytes_backward": tcp_b[0].window if tcp_b else -1,
        "act_data_pkt_fwd": sum(1 for p in fw if p.payload > 0),
        "min_seg_size_forward": min((p.hdr_len for p in tcp_f), default=0) if tcp_f else (fw[0].hdr_len if fw else 0),
    }
    return np.array([v[n] for n in FEATURES], dtype=np.float64)
