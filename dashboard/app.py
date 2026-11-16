"""Local Edge-AI IDS dashboard.   streamlit run dashboard/app.py"""
import glob, json, os, signal, subprocess, sys, time
import pandas as pd
import psutil
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results"); STATE = os.path.join(RES, "live_state.json")
PY = sys.executable; MON = os.path.join(ROOT, "python_network", "network_monitor.py")

st.set_page_config(page_title="Edge AI IDS", page_icon="🛡️", layout="wide")

def load_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def monitor_alive():
    pid = st.session_state.get("mon_pid")
    return bool(pid) and psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE

# ---------------- sidebar: control
with st.sidebar:
    st.title("🛡️ Edge AI IDS")
    st.caption("All inference runs locally (C++ MLP engine). No cloud calls.")
    mode = st.radio("Traffic source", ["Dataset replay (real test flows)", "PCAP replay", "Live interface"])
    if mode.startswith("Dataset"):
        nflows = st.number_input("Flows to replay", 100, 100000, 3000, 100); delay = st.number_input("Delay per flow (s)", 0.0, 1.0, 0.002, 0.001, format="%.3f")
        st.caption("Held-out test flows (30% attacks) with ground truth.")
    elif mode == "PCAP replay":
        pcaps = sorted(glob.glob(os.path.join(ROOT, "datasets", "pcaps", "*.pcap*")))
        pcap = st.selectbox("PCAP file", pcaps) if pcaps else st.text_input("PCAP path")
        speed = st.number_input("Replay speed (0 = max)", 0.0, 1000.0, 0.0)
    else:
        iface = st.text_input("Interface", "lo"); st.caption("Needs root or CAP_NET_RAW.")
    thr = st.slider("Attack threshold", 0.05, 0.99, 0.5, 0.01)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", disabled=monitor_alive(), use_container_width=True):
        for f in (STATE, os.path.join(RES, "alerts.jsonl")):
            if os.path.exists(f): os.remove(f)
        args = [PY, MON, "--threshold", str(thr)] + (["--dataset-replay", str(int(nflows)), "--delay", str(delay)] if mode.startswith("Dataset") else ["--pcap", pcap, "--speed", str(speed)] if mode == "PCAP replay" else ["--iface", iface])
        st.session_state.mon_pid = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(os.path.join(RES, "monitor_stderr.log"), "w")).pid
    if c2.button("■ Stop", disabled=not monitor_alive(), use_container_width=True):
        os.kill(st.session_state.mon_pid, signal.SIGINT)
    st.markdown(f"**Monitor process:** {'🟢 running' if monitor_alive() else '⚪ not running'}")
    refresh = st.slider("Refresh (s)", 1, 5, 1)

st.title("Network Intrusion Detection — Edge AI")
tab_live, tab_perf, tab_exp = st.tabs(["Live monitoring", "Model performance", "Experiments"])

@st.fragment(run_every=refresh)
def live():
    s = load_json(STATE)
    if not s:
        st.info("No monitor data yet. Choose a source and press **Start** (or run network_monitor.py)."); return
    stale = time.time() - s["updated"] > 5 and s["status"] == "running"
    a = st.columns(5)
    a[0].metric("System", "STALE" if stale else s["status"].upper()); a[1].metric("Capture", s["capture_state"])
    a[2].metric("Packets", f"{s['packets']:,}"); a[3].metric("Flows classified", f"{s['flows']:,}"); a[4].metric("Active flows", s["active_flows"])
    b = st.columns(5)
    b[0].metric("Normal", f"{s['normal']:,}"); b[1].metric("Suspicious", f"{s['suspicious']:,}")
    b[2].metric("Latency mean/p95 (ms)", f"{s['latency_ms']['mean']:.2f} / {s['latency_ms']['p95']:.2f}")
    b[3].metric("Process CPU %", f"{s['cpu_percent']:.0f}"); b[4].metric("Process RAM (MB)", f"{s['rss_mb']:.0f}")
    if s.get("truth_agreement") is not None: st.metric("Agreement with ground truth (replayed dataset flows)", f"{s['truth_agreement']:.4f}", help=f"{s['truth_checked']:,} flows checked")
    if s.get("capture_error"): st.error(s["capture_error"])
    l, r = st.columns([1, 2])
    with l:
        st.subheader("Traffic by category")
        if s["by_category"]: st.bar_chart(pd.Series(s["by_category"]).sort_values(ascending=False))
        st.caption(f"Throughput: {s['throughput']['packets_per_s']:.0f} pkt/s, {s['throughput']['flows_per_s']:.1f} flows/s · threshold {s['threshold']}")
    with r:
        st.subheader("Recent security alerts")
        if s["alerts"]: st.dataframe(pd.DataFrame(s["alerts"]), use_container_width=True, hide_index=True, height=360)
        else: st.success("No alerts.")
live()

with tab_perf:
    ev = load_json(os.path.join(RES, "final_evaluation.json"))
    if not ev: st.warning("Run evaluation/train_final.py to generate results/final_evaluation.json")
    else:
        for task in ("binary", "multiclass"):
            r = ev[task]; st.subheader(f"{task.capitalize()} model — held-out test set ({sum(map(sum, r['confusion_matrix'])):,} flows)")
            cols = st.columns(5 if task == "binary" else 4)
            cols[0].metric("Accuracy", f"{r['accuracy']:.4f}")
            if task == "binary":
                cols[1].metric("Precision", f"{r['precision']:.4f}"); cols[2].metric("Recall", f"{r['recall']:.4f}"); cols[3].metric("FPR", f"{r['false_positive_rate']:.4f}"); cols[4].metric("FNR", f"{r['false_negative_rate']:.4f}")
            else:
                cols[1].metric("Macro precision", f"{r['macro_precision']:.4f}"); cols[2].metric("Macro recall", f"{r['macro_recall']:.4f}"); cols[3].metric("Macro F1", f"{r['macro_f1']:.4f}")
            st.dataframe(pd.DataFrame(r["per_class"]).T, use_container_width=True)
            st.write("Confusion matrix (rows = true, cols = predicted)")
            st.dataframe(pd.DataFrame(r["confusion_matrix"], index=r["classes"], columns=r["classes"]), use_container_width=True)
        edge = load_json(os.path.join(RES, "edge_benchmark.json"))
        if edge: st.subheader("Edge benchmark (this laptop)"); st.json(edge)

with tab_exp:
    ex = load_json(os.path.join(RES, "experiments.json"))
    if ex:
        st.subheader("Baselines (binary, test)")
        st.dataframe(pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")} for k, v in ex["baselines"].items()}).T)
        st.subheader("Neural-network experiments (binary, test)")
        st.dataframe(pd.DataFrame(ex["experiments"]).drop(columns=["params"]), use_container_width=True, hide_index=True)
        if "unseen_family_generalisation" in ex: st.subheader("Unseen-attack-family generalisation"); st.json(ex["unseen_family_generalisation"])
    else: st.info("experiments.json not found")
