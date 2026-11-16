"""Edge benchmark on THIS machine: latency, throughput, CPU, RAM, model size, and offline check."""
import json, os, socket, sys, time
import numpy as np, psutil
from common import *
from preprocessing import FEATURES
from packet_capture import CaptureStatus, pcap_source
from flow_manager import FlowManager
from feature_extractor import extract
from network_monitor import EdgeDetector

def offline_check():
    """True if no route to the internet is reachable (so successful inference proves local-only operation)."""
    try: socket.create_connection(("1.1.1.1", 53), timeout=1.5).close(); return False
    except OSError: return True

proc = psutil.Process(); det = EdgeDetector(); out = {"offline_during_run": offline_check(), "cpu_count": psutil.cpu_count()}
rss0 = proc.memory_info().rss / 2**20
# 1. real preprocessed test flows from the dataset: model-only latency
Xraw = np.load(os.path.join(PROC, "raw_split.npz"), allow_pickle=True)["X"]; sp = np.load(os.path.join(PROC, "raw_split.npz"), allow_pickle=True)["split"]
X = Xraw[sp == 2][:20000].astype(np.float64); del Xraw
det.classify_vectors(X[:10])  # warm-up
lat = []
for i in range(2000):
    t = time.perf_counter(); det.classify_vectors(X[i:i + 1]); lat.append((time.perf_counter() - t) * 1000)
lat = np.array(lat); out["single_flow_pipeline_ms"] = {"description": "preprocess + binary MLP + multiclass MLP (C++ via ctypes), batch=1", "mean": lat.mean(), "p50": np.percentile(lat, 50), "p95": np.percentile(lat, 95), "p99": np.percentile(lat, 99)}
for bs in (1, 64, 1024):
    n = 20000 // bs * bs; t = time.perf_counter(); cpu0 = proc.cpu_times()
    for i in range(0, n, bs): det.classify_vectors(X[i:i + bs])
    dt = time.perf_counter() - t; out[f"throughput_batch{bs}_flows_per_s"] = n / dt
# 2. packet path: capture(pcap) -> flows -> features -> inference, on the synthetic mixed fixture
st = CaptureStatus(); fm = FlowManager(); flows = []; t = time.perf_counter(); cpu0 = proc.cpu_times(); npk = 0
for p in pcap_source(os.path.join(ROOT, "datasets", "pcaps", "synthetic_mixed.pcap"), st): flows += fm.add(p); npk += 1
flows += fm.flush(); t_parse = time.perf_counter() - t
t = time.perf_counter(); V = np.stack([extract(f) for f in flows]); t_feat = time.perf_counter() - t
t = time.perf_counter(); det.classify_vectors(V); t_inf = time.perf_counter() - t
cpu = proc.cpu_times(); out["packet_path_synthetic_mixed"] = {"packets": npk, "flows": len(flows), "pcap_parse_and_flow_s": t_parse, "feature_extraction_s": t_feat, "inference_s": t_inf,
    "packets_per_s_end_to_end": npk / (t_parse + t_feat + t_inf), "feature_extraction_ms_per_flow": t_feat / len(flows) * 1000, "inference_ms_per_flow_batched": t_inf / len(flows) * 1000,
    "note": "most time is scapy PCAP decoding; live capture rate is bounded by scapy, not by the MLP"}
out["cpu_seconds_user+sys_total"] = cpu.user + cpu.system; out["rss_mb_start"] = rss0; out["rss_mb_end"] = proc.memory_info().rss / 2**20
out["model_size_bytes"] = {"binary": det.bin.size_bytes, "multiclass": det.multi.size_bytes}
json.dump(out, open(os.path.join(ROOT, "results", "edge_benchmark.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
