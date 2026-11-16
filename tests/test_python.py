import json, os, struct, subprocess, sys, tempfile
import numpy as np
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("python_network", "integration", "evaluation"): sys.path.insert(0, os.path.join(ROOT, sub))
from preprocessing import FEATURES, Preprocessor, clean_label, MULTI_CLASSES
from packet_capture import CaptureStatus, pcap_source, PacketRecord, FIN, SYN, ACK, RST
from flow_manager import FlowManager
from feature_extractor import extract
from engine import NativeModel, EngineError
MODELS = os.path.join(ROOT, "models"); PCAPS = os.path.join(ROOT, "datasets", "pcaps")

def rec(ts, src, dst, sp, dp, payload=0, flags=ACK, proto=6, win=1000, hdr=20):
    return PacketRecord(ts, src, dst, sp, dp, proto, payload + hdr + 20, payload, hdr, flags, win)

# ---------------- preprocessing
def test_clean_label_mojibake():
    assert clean_label("Web Attack ï¿½ Brute Force") == "Web Attack - Brute Force"
    assert clean_label(" BENIGN ") == "BENIGN"

def test_preprocessor_fit_uses_only_given_data_and_handles_bad_values():
    rng = np.random.default_rng(0); X = rng.lognormal(3, 2, (500, len(FEATURES))); X[3, 5] = np.inf; X[4, 6] = np.nan; X[:, 10] = 7
    p = Preprocessor().fit(X); Z = p.transform(X)
    assert Z.shape == X.shape and Z.dtype == np.float32 and np.isfinite(Z).all() and np.abs(Z).max() <= 10
    assert np.allclose(Z[:, 10], 0)                                  # constant column -> 0, no division by zero
    assert abs(Z[:, 0].mean()) < 0.05 and abs(Z[:, 0].std() - 1) < 0.05
    test = rng.lognormal(10, 3, (10, len(FEATURES)))                  # different distribution: stats must NOT change
    before = p.mean.copy(); p.transform(test); assert np.array_equal(before, p.mean)
    with pytest.raises(ValueError): p.transform(X[:, :5])

def test_preprocessor_roundtrip(tmp_path):
    X = np.random.default_rng(1).lognormal(2, 1, (200, len(FEATURES))); p = Preprocessor().fit(X); f = tmp_path / "p.json"; p.save(str(f))
    assert np.array_equal(p.transform(X), Preprocessor.load(str(f)).transform(X))
    sub = p.select(FEATURES[1:]); assert np.array_equal(sub.transform(X[:, 1:]), p.transform(X)[:, 1:])

def test_deployed_preprocessing_matches_training_pipeline():
    """Rows from the real dataset, run through the DEPLOYED json, must equal the preprocessed arrays used in training."""
    from common import read_xy, PROC
    S = np.load(os.path.join(PROC, "raw_split.npz"), allow_pickle=True)
    full = Preprocessor.load(os.path.join(PROC, "preprocessing_full.json"))
    Xtr, _ = read_xy(os.path.join(PROC, "binary_train")); raw = S["X"][S["split"] == 0][:2000]
    assert np.allclose(full.transform(raw), Xtr[:2000], atol=1e-5)
    # leakage guard: train/val/test row sets are disjoint (dedup before split)
    X = S["X"]; sp = S["split"]; h = [set(map(bytes, np.ascontiguousarray(X[sp == i][:20000]).view(np.uint8).reshape(-1, X.shape[1] * 4))) for i in (0, 1, 2)]
    assert not (h[0] & h[2]) and not (h[0] & h[1])

# ---------------- flows
def test_flow_direction_counts_and_fin_close():
    fm = FlowManager(); t = 100.0; done = []
    seq = [("a", "b", 1111, 80, SYN, 0), ("b", "a", 80, 1111, SYN | ACK, 0), ("a", "b", 1111, 80, ACK, 0), ("a", "b", 1111, 80, ACK, 100),
           ("b", "a", 80, 1111, ACK, 500), ("a", "b", 1111, 80, FIN | ACK, 0), ("b", "a", 80, 1111, FIN | ACK, 0)]
    for i, (s, d, sp, dp, fl, pl) in enumerate(seq): done += fm.add(rec(t + i * .1, s, d, sp, dp, pl, fl))
    assert done == [] and len(fm.flows) == 1                      # waits for the final ACK
    done += fm.add(rec(t + .8, "a", "b", 1111, 80, 0, ACK)); assert len(done) == 1 and not fm.flows
    f = done[0]; assert (len(f.fwd), len(f.bwd)) == (5, 3) and f.dport == 80
    v = dict(zip(FEATURES, extract(f)))
    assert v["Total Length of Fwd Packets"] == 100 and v["Total Length of Bwd Packets"] == 500 and v["Destination Port"] == 80
    assert v["Flow Duration"] == pytest.approx(800000) and v["FIN Flag Count"] == 2 and v["SYN Flag Count"] == 2 and v["act_data_pkt_fwd"] == 1
    assert v["Init_Win_bytes_forward"] == 1000 and v["Fwd Header Length"] == 100 and v["Flow Bytes/s"] == pytest.approx(600 / 0.8)
    assert v["Fwd Packet Length Max"] == 100 and v["Bwd Packet Length Mean"] == pytest.approx(500 / 3)

def test_flow_idle_timeout_and_rst():
    fm = FlowManager(idle_timeout=5)
    fm.add(rec(0, "a", "b", 1, 2)); assert fm.expire(3) == [] and len(fm.expire(10)) == 1
    fm.add(rec(20, "a", "b", 1, 2)); out = fm.add(rec(21, "b", "a", 2, 1, flags=RST)); assert len(out) == 1
    fm.add(rec(30, "a", "b", 1, 2)); out = fm.add(rec(100, "a", "b", 1, 2)); assert len(out) == 1 and len(fm.flows) == 1   # same tuple after timeout = new flow

def test_extract_shape_and_udp_no_bwd():
    fm = FlowManager(); fm.add(rec(0, "a", "b", 5000, 53, 30, 0, 17, 0, 8)); f = fm.flush()[0]; v = extract(f)
    assert v.shape == (len(FEATURES),) and v[FEATURES.index("Init_Win_bytes_forward")] == -1 and v[FEATURES.index("Total Backward Packets")] == 0
    assert np.isinf(v[FEATURES.index("Flow Bytes/s")])             # zero-duration flow -> inf, mapped to median by the Preprocessor
    assert np.isfinite(Preprocessor().fit(np.random.default_rng(0).lognormal(1, 1, (50, len(FEATURES)))).transform(v)).all()

def test_pcap_flow_generation():
    for name, n in (("normal", 40), ("portscan", 399), ("flood", 300)):
        fm = FlowManager(); st = CaptureStatus(); flows = []
        for p in pcap_source(os.path.join(PCAPS, f"synthetic_{name}.pcap"), st): flows += fm.add(p)
        flows += fm.flush(); assert len(flows) == n and st.state == "finished" and st.error == ""

# ---------------- python <-> C++
def numpy_forward(path, X):
    with open(path, "rb") as f:
        assert f.read(8) == b"IDSMLP01"; n = struct.unpack("<I", f.read(4))[0]; sizes = struct.unpack(f"<{n}Q", f.read(8 * n))
        a = X.astype(np.float64)
        for l in range(n - 1):
            W = np.frombuffer(f.read(4 * sizes[l] * sizes[l + 1]), "<f4").reshape(sizes[l], sizes[l + 1]).astype(np.float64)
            b = np.frombuffer(f.read(4 * sizes[l + 1]), "<f4").astype(np.float64); a = a @ W + b
            if l < n - 2: a = np.maximum(a, 0)
        a = np.exp(a - a.max(1, keepdims=True)); return a / a.sum(1, keepdims=True)

@pytest.mark.parametrize("fn", ["binary_model.bin", "multiclass_model.bin"])
def test_cpp_engine_matches_independent_numpy_forward(fn):
    m = NativeModel(os.path.join(MODELS, fn)); X = np.random.default_rng(2).normal(0, 1, (64, m.n_in)).astype(np.float32)
    P = m.predict_proba(X); assert P.shape == (64, m.n_out) and np.allclose(P.sum(1), 1, atol=1e-5)
    assert np.allclose(P, numpy_forward(os.path.join(MODELS, fn), X), atol=1e-4)

def test_engine_input_validation_and_errors(tmp_path):
    m = NativeModel(os.path.join(MODELS, "binary_model.bin"))
    with pytest.raises(ValueError): m.predict_proba(np.zeros((1, m.n_in + 1)))
    with pytest.raises(ValueError): m.predict_proba(np.full((1, m.n_in), np.nan))
    with pytest.raises(EngineError): NativeModel(str(tmp_path / "missing.bin"))
    bad = tmp_path / "bad.bin"; bad.write_bytes(b"not a model"); 
    with pytest.raises(EngineError): NativeModel(str(bad))

def test_trained_model_predicts_real_test_data():
    from common import read_xy, PROC
    from metrics import report
    meta = json.load(open(os.path.join(MODELS, "preprocessing.json"))); cfg = meta["training_config"]
    X, y = read_xy(os.path.join(PROC, "binary_test")); idx = [FEATURES.index(f) for f in meta["features"]]
    n = 20000; r = report(y[:n], NativeModel(os.path.join(MODELS, "binary_model.bin")).predict(X[:n][:, idx]), ["BENIGN", "ATTACK"])
    assert r["accuracy"] > 0.98 and r["recall"] > 0.9

# ---------------- monitor end-to-end + dashboard
def test_monitor_end_to_end_offline(tmp_path):
    from network_monitor import EdgeDetector, Monitor
    mon = Monitor(EdgeDetector(), state_path=str(tmp_path / "state.json")); st = mon.status
    mon.run(pcap_source(os.path.join(PCAPS, "synthetic_mixed.pcap"), st))
    s = json.load(open(tmp_path / "state.json"))
    assert s["packets"] == 1688 and s["flows"] == 739 and s["normal"] + s["suspicious"] == s["flows"] and s["status"] == "stopped"
    assert s["latency_ms"]["mean"] > 0 and s["rss_mb"] > 0 and sum(s["by_category"].values()) == s["flows"]
    for a in s["alerts"]: assert {"time", "src", "dst", "dport", "category", "confidence"} <= set(a)

def test_dashboard_renders_with_real_state(tmp_path):
    from streamlit.testing.v1 import AppTest
    state = os.path.join(ROOT, "results", "live_state.json")
    subprocess.run([sys.executable, os.path.join(ROOT, "python_network", "network_monitor.py"), "--pcap", os.path.join(PCAPS, "synthetic_mixed.pcap")], check=True, capture_output=True)
    at = AppTest.from_file(os.path.join(ROOT, "dashboard", "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}; assert {"Packets", "Flows classified", "Suspicious", "Normal"} <= labels
    vals = {m.label: m.value for m in at.metric}; assert vals["Packets"].replace(",", "") == "1688"

def test_dataset_replay_detects_real_attacks(tmp_path):
    from network_monitor import EdgeDetector, Monitor
    d = np.load(os.path.join(ROOT, "datasets", "processed", "raw_split.npz"), allow_pickle=True); m = d["split"] == 2
    X, lab = d["X"][m][:60000].astype(np.float64), d["lab"][m][:60000]
    rng = np.random.default_rng(0); idx = np.concatenate([rng.choice(np.flatnonzero(lab != "BENIGN"), 300, replace=False), rng.choice(np.flatnonzero(lab == "BENIGN"), 700, replace=False)])
    mon = Monitor(EdgeDetector(), state_path=str(tmp_path / "s.json")); mon.run_dataset(X[idx], lab[idx])
    s = json.load(open(tmp_path / "s.json")); assert s["flows"] == 1000 and s["truth_checked"] == 1000 and s["truth_agreement"] > 0.98 and s["suspicious"] > 250
