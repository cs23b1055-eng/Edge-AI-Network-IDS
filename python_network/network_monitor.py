"""Edge AI monitor: capture -> flows -> features -> preprocess -> C++ MLP -> alerts + live state.
Everything runs locally; no network access is needed for inference.

  python network_monitor.py --pcap file.pcap [--speed 0]      # offline replay
  sudo python network_monitor.py --iface eth0                  # live capture (needs CAP_NET_RAW)
"""
import argparse, json, os, sys, threading, time
from collections import deque
import numpy as np
import psutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "integration"))
from engine import NativeModel
from feature_extractor import extract
from flow_manager import FlowManager
from packet_capture import CaptureStatus, live_source, pcap_source
from preprocessing import FEATURES, Preprocessor

STATE_PATH = os.path.join(ROOT, "results", "live_state.json")

class EdgeDetector:
    """Loads artefacts once; classify_flow() is pure inference (no training, no network)."""
    def __init__(self, models_dir=os.path.join(ROOT, "models"), threshold=0.5):
        self.pre = Preprocessor.load(os.path.join(models_dir, "preprocessing.json"))
        meta = json.load(open(os.path.join(models_dir, "preprocessing.json")))
        self.multi_classes = meta["multi_classes"]
        self.bin = NativeModel(os.path.join(models_dir, "binary_model.bin"))
        self.multi = NativeModel(os.path.join(models_dir, "multiclass_model.bin"))
        self.threshold = threshold
        self.col_idx = [FEATURES.index(f) for f in self.pre.features]   # model may use a subset of the 57 extracted features
        if self.bin.n_in != len(self.pre.features) or self.multi.n_in != len(self.pre.features):
            raise RuntimeError("model input size does not match the preprocessing feature list")

    def classify_vectors(self, raw):
        """raw: (n, 57) matrix in FEATURES order -> list of dicts."""
        raw = np.atleast_2d(raw)
        if raw.shape[1] != len(FEATURES): raise ValueError(f"expected {len(FEATURES)} raw features")
        Z = self.pre.transform(raw[:, self.col_idx])
        pb = self.bin.predict_proba(Z)[:, 1]; pm = self.multi.predict_proba(Z)
        out = []
        for i in range(len(Z)):
            attack = bool(pb[i] >= self.threshold); mc = int(pm[i].argmax())
            if attack:
                cat = self.multi_classes[mc] if mc != 0 else "Unclassified attack"
                conf = float(pb[i])
            else:
                cat, conf = "BENIGN", float(1 - pb[i])
            out.append({"malicious": attack, "category": cat, "confidence": conf, "p_attack": float(pb[i]),
                        "category_probs": {c: float(p) for c, p in zip(self.multi_classes, pm[i])}})
        return out

class Monitor:
    def __init__(self, detector, idle_timeout=30.0, state_path=STATE_PATH, alert_log=None):
        self.det = detector; self.fm = FlowManager(idle_timeout=idle_timeout)
        self.status = CaptureStatus(); self.state_path = state_path
        self.alert_log = alert_log or os.path.join(os.path.dirname(state_path), "alerts.jsonl")
        self.proc = psutil.Process(); self.proc.cpu_percent(None)
        self.packets = 0; self.flows = 0; self.normal = 0; self.suspicious = 0
        self.by_cat = {}; self.alerts = deque(maxlen=100); self.lat_ms = deque(maxlen=2000)
        self.t0 = time.time(); self.last_write = 0.0; self.running = True; self.proc_seconds = 0.0
        self.truth_n = 0; self.truth_ok = 0
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

    def _record(self, res, dt, meta):
        self.lat_ms.append(dt); self.proc_seconds += dt / 1000
        self.flows += 1; self.by_cat[res["category"]] = self.by_cat.get(res["category"], 0) + 1
        if "truth_malicious" in meta:
            self.truth_n += 1; self.truth_ok += int(meta["truth_malicious"] == res["malicious"])
        if res["malicious"]:
            self.suspicious += 1
            a = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "src": meta["src"], "sport": meta["sport"], "dst": meta["dst"], "dport": meta["dport"],
                 "proto": meta["proto"], "category": res["category"], "confidence": round(res["confidence"], 4),
                 "packets": meta["packets"], "latency_ms": round(dt, 3)}
            if "truth" in meta: a["true_label"] = meta["truth"]
            self.alerts.appendleft(a)
            with open(self.alert_log, "a") as fh: fh.write(json.dumps(a) + "\n")
        else:
            self.normal += 1

    def _handle_flows(self, flows):
        for f in flows:
            t = time.perf_counter()
            res = self.det.classify_vectors(extract(f)[None, :])[0]
            dt = (time.perf_counter() - t) * 1000
            self._record(res, dt, {"src": f.src, "sport": f.sport, "dst": f.dst, "dport": f.dport,
                                   "proto": "TCP" if f.proto == 6 else "UDP", "packets": f.n_packets})

    def run_dataset(self, X, truth_labels, delay=0.0):
        """Replay REAL labelled flow-feature rows (held-out test split) through the same detector.
        Demonstrates detection on genuine attack flows; ground truth is shown next to each alert."""
        self.status.state = "replaying"
        try:
            for i in range(len(X)):
                t = time.perf_counter(); res = self.det.classify_vectors(X[i:i + 1])[0]; dt = (time.perf_counter() - t) * 1000
                npk = int(X[i, FEATURES.index("Total Fwd Packets")] + X[i, FEATURES.index("Total Backward Packets")]); self.packets += npk
                self._record(res, dt, {"src": "dataset", "sport": 0, "dst": "dataset", "dport": int(X[i, 0]), "proto": "n/a", "packets": npk,
                                       "truth": str(truth_labels[i]), "truth_malicious": str(truth_labels[i]) != "BENIGN"})
                self._write()
                if delay: time.sleep(delay)
            self.status.state = "finished"
        finally:
            self.running = False; self._write(force=True)

    def snapshot(self):
        lat = np.array(self.lat_ms) if self.lat_ms else np.zeros(1)
        el = max(time.time() - self.t0, 1e-9); mi = self.proc.memory_info()
        return {"updated": time.time(), "status": "running" if self.running else "stopped", "capture_state": self.status.state,
                "capture_error": self.status.error, "packets": self.packets, "packets_ignored": self.status.ignored,
                "active_flows": len(self.fm.flows), "flows": self.flows, "normal": self.normal, "suspicious": self.suspicious,
                "by_category": self.by_cat, "alerts": list(self.alerts)[:50],
                "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
                "throughput": {"packets_per_s": self.packets / el, "flows_per_s": self.flows / el},
                "cpu_percent": self.proc.cpu_percent(None), "rss_mb": mi.rss / 2**20, "system_cpu_percent": psutil.cpu_percent(None),
                "truth_checked": self.truth_n, "truth_agreement": (self.truth_ok / self.truth_n) if self.truth_n else None, "uptime_s": el, "threshold": self.det.threshold}

    def _write(self, force=False):
        if not force and time.time() - self.last_write < 0.5: return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh: json.dump(self.snapshot(), fh)
        os.replace(tmp, self.state_path); self.last_write = time.time()

    def run(self, source):
        try:
            last_ts = 0.0
            for p in source:
                self.packets += 1; last_ts = p.ts
                self._handle_flows(self.fm.add(p))
                if self.packets % 500 == 0:
                    self._handle_flows(self.fm.expire(last_ts if self.status.state == "replaying" else time.time()))
                self._write()
            self._handle_flows(self.fm.flush())
        finally:
            self.running = False; self._write(force=True)

def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcap"); g.add_argument("--iface"); g.add_argument("--dataset-replay", type=int, metavar="N", help="replay N real held-out test flows (30%% attacks) from the dataset")
    ap.add_argument("--speed", type=float, default=0.0, help="PCAP replay speed (0 = as fast as possible)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between flows in --dataset-replay"); ap.add_argument("--idle-timeout", type=float, default=30.0); ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    det = EdgeDetector(threshold=a.threshold); mon = Monitor(det, idle_timeout=a.idle_timeout); stop = threading.Event()
    if a.dataset_replay:
        d = np.load(os.path.join(ROOT, "datasets", "processed", "raw_split.npz"), allow_pickle=True); m = d["split"] == 2
        X, lab = d["X"][m], d["lab"][m]; rng = np.random.default_rng(int(time.time())); atk = np.flatnonzero(lab != "BENIGN"); ben = np.flatnonzero(lab == "BENIGN")
        n = a.dataset_replay; idx = np.concatenate([rng.choice(atk, int(n * .3), replace=False), rng.choice(ben, n - int(n * .3), replace=False)]); rng.shuffle(idx)
        mon.run_dataset(X[idx].astype(np.float64), lab[idx], a.delay)
        s = mon.snapshot(); print(json.dumps({k: s[k] for k in ("flows", "normal", "suspicious", "by_category", "truth_checked", "truth_agreement", "latency_ms")}, indent=1)); return
    src = pcap_source(a.pcap, mon.status, a.speed, stop) if a.pcap else live_source(a.iface, mon.status, stop=stop)
    try: mon.run(src)
    except KeyboardInterrupt: stop.set()
    s = mon.snapshot(); print(json.dumps({k: s[k] for k in ("packets", "flows", "normal", "suspicious", "by_category", "latency_ms", "throughput", "rss_mb", "capture_state", "capture_error")}, indent=1))

if __name__ == "__main__":
    main()
