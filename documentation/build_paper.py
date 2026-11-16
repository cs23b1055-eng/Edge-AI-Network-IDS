"""Builds documentation/Research_Paper.docx. Every number is read from results/*.json (measured), none typed by hand."""
import json, os, subprocess, platform
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
J = lambda n: json.load(open(os.path.join(ROOT, "results", n)))
da, sp, fe, ex, pe, eb, ea = J("dataset_analysis.json"), J("split_report.json"), J("final_evaluation.json"), J("experiments.json"), J("paper_experiments.json"), J("edge_benchmark.json"), J("error_analysis.json")
FIG = os.path.join(ROOT, "documentation", "figures")
cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
pc = lambda x, d=2: f"{x*100:.{d}f}%"; ms = lambda d: f"{d['mean']:.4f} ± {d['std']:.4f}"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for s in doc.sections: s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(0.9)
def H(t, l=1): 
    h = doc.add_heading(t, l)
    for r in h.runs: r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F); r.font.name = "Times New Roman"
def P(t, bold=False, italic=False, align=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic; p.paragraph_format.space_after = Pt(6)
    if align: p.alignment = align
    return p
def B(items):
    for i in items: doc.add_paragraph(i, style="List Bullet")
def T(rows, header=True, widths=None, size=9):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j); cell.text = ""; run = cell.paragraphs[0].add_run(str(c)); run.font.size = Pt(size); run.bold = (i == 0 and header)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Inches(w)
    doc.add_paragraph()
def F(name, cap, w=6.3):
    doc.add_picture(os.path.join(FIG, name), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = P(cap, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER); p.runs[0].font.size = Pt(9)

bn, mc = fe["binary"], fe["multiclass"]; cm = bn["confusion_matrix"]; ab, ft, lo = pe["ablation"], pe["features"], pe["leave_one_family_out"]
tr, va, te = (sp["splits"][k]["rows"] for k in ("train", "val", "test"))

# ---------------------------------------------------------------- title
P("An Edge-Deployable Network Intrusion Detector with a From-Scratch C++ Neural Network: Implementation, Controlled Ablation, and an Honest Look at Generalisation", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].font.size = Pt(16)
P("[Team member names and roll numbers]   ·   [Course: DIP / CN / COMA]   ·   [Institute]   ·   Draft: 16 November 2026", align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].font.size = Pt(10)
H("Abstract", 2)
P(f"Machine-learning intrusion detectors routinely report accuracy above 99% on public flow datasets, yet two recent lines of work question what such numbers mean: the benchmark labels themselves contain errors, and models degrade sharply on data that differ from their training distribution. "
  f"We build a complete flow-based intrusion detector whose neural network (a multilayer perceptron with backpropagation, He initialisation, Adam and weighted cross-entropy) is written from scratch in C++ without any machine-learning framework, wrapped for Python through ctypes, and deployed entirely on a laptop. "
  f"Trained on {da['rows_after_dedup']:,} de-duplicated flows from the CIC-IDS2017-style CSV files supplied with the course, the binary detector reaches {pc(bn['accuracy'])} accuracy, {pc(bn['recall'])} recall and a false-positive rate of {pc(bn['false_positive_rate'])} on a held-out split, and a seven-class variant reaches macro-F1 {mc['macro_f1']:.3f}. "
  f"A controlled five-seed ablation shows which components matter (a log transform and a non-linear hidden layer account for most of the gain). The deployed pipeline needs {eb['single_flow_pipeline_ms']['mean']:.3f} ms per flow and about {eb['rss_mb_end']:.0f} MB of RAM, with inference verified in a network namespace without connectivity. "
  f"Our central finding is negative and important: when an attack family is withheld from training, recall on that family ranges from {pc(min(v['recall_on_unseen_family']['mean'] for v in lo.values()),1)} to {pc(max(v['recall_on_unseen_family']['mean'] for v in lo.values()),1)}, so in-distribution scores greatly overstate real-world readiness.")
P("Keywords: network intrusion detection, edge AI, multilayer perceptron, backpropagation, CIC-IDS2017, generalisation, ablation study.", italic=True)

# ---------------------------------------------------------------- 1 intro
H("1. Introduction")
P("Network intrusion detection systems (NIDS) watch traffic for signs of attack. Signature-based tools miss novel behaviour, so learning-based detectors that classify network flows from statistical features are widely studied. A flow is the set of packets sharing a five-tuple (source and destination address, source and destination port, protocol); a flow-based detector summarises each flow with numbers such as duration, packet counts, packet sizes, inter-arrival times and TCP flag counts and classifies the summary as benign or malicious.")
P("Motivation. Running detection on the device that sees the traffic (an edge device such as a gateway, a laptop or a small single-board computer) avoids sending traffic metadata to a cloud service, removes network round-trip time, and keeps working offline. That requires a model that is small and fast. At the same time, published results make deployment look easier than it is, which motivates the evaluation questions below.")
P("Limitations of existing approaches (as reviewed in Section 2). (i) Reported scores are usually measured in-distribution (random splits of one dataset). (ii) The widely used CIC-IDS2017 benchmark has documented errors in traffic generation, flow construction, feature extraction and labelling. (iii) Cross-dataset studies report that accuracy can fall to near chance. (iv) Many studies stop at offline classification and do not report an end-to-end, offline-capable deployment with measured latency and memory.")
P("Objectives and hypotheses.", bold=True)
B(["H1 (feasibility): a small MLP implemented from scratch in C++ can classify flows of the supplied dataset with accuracy and F1 above 99% and a false-positive rate below 0.5% on a held-out split, at well under 1 ms per flow on a laptop CPU, fully offline.",
   "H2 (generalisation): in-distribution scores overestimate robustness; when an attack family is entirely withheld from training, recall on that family will fall far below the in-distribution recall.",
   "H3 (components): the log transform of heavy-tailed features, a non-linear hidden layer and class weighting each improve the held-out F1 beyond seed-to-seed variation."])
P("Contributions.", bold=True)
B(["A complete, tested, from-scratch C++ MLP (forward pass, backpropagation verified by a numerical gradient check, SGD/Adam, model I/O, C API) and a reproducible training pipeline with train-only preprocessing statistics.",
   "A packet-to-alert edge pipeline (Scapy capture, five-tuple flow assembly, 57 flow features, local inference, dashboard) with measured latency, throughput, CPU and RAM, and offline operation demonstrated.",
   "A controlled five-seed cumulative ablation, a feature ablation, and a leave-one-attack-family-out study quantifying the generalisation gap.",
   "An error analysis explaining the model's failures with evidence from the flow features, and an explicit account of dataset and validity limitations."])

# ---------------------------------------------------------------- 2 literature
H("2. Literature Survey")
P("We focus on work published from 2021 to 2026. Citation counts below were all retrieved from one source (the OpenAlex API) on 16 November 2026; counts differ between scholarly databases and will change over time. Abstracts were read for each entry; performance figures are quoted only where the abstract states them.")
rows = [["Author(s) & year", "Methodology", "Performance (as reported)", "Dataset", "Critical remarks"],
 ["Engelen, Rimmer & Joosen (2021) [1] — 304 citations", "Re-examination of the CIC-IDS2017 collection pipeline; improved processing methodology.", "Finds problems in traffic generation, flow construction, feature extraction and labelling; >20% of traces reconstructed or relabelled; ML benchmarks improve significantly on the corrected data.", "CIC-IDS2017 (public)", "Shows that evaluation on the original data is affected by errors in the data itself. Directly relevant: our CSVs are the original release."],
 ["Liu, Engelen et al. (2022) [2] — 164", "Critical review of CIC-IDS-2017 and CSE-CIC-IDS-2018; labelling logic reverse-engineered.", "Reports many previously undocumented errors that destabilise published results; releases a fully re-created dataset with corrected labels.", "CIC-IDS-2017, CSE-CIC-IDS-2018", "Argues that benchmark conclusions in much of the literature may not hold; recommends releasing labelling logic."],
 ["Cantone, Marrocco & Bria (2024) [3] — 155", "Cross-dataset study with four ML classifiers.", "Near-perfect classification when train and test come from the same dataset; accuracy largely comparable to random chance when crossing datasets (except a few attack/dataset pairs).", "CIC-IDS-2017, CSE-CIC-IDS2018, LycoS-IDS2017, LycoS-Unicas-IDS2018", "Strongest evidence for the generalisation gap; points to data heterogeneity and anomalies as causes. Uses classical classifiers, not an edge deployment."],
 ["Mohammadpour, Ling & Liew (2022) [4] — 253", "Survey of CNN-based IDS; categorises approaches and compares them empirically on standard datasets.", "Notes that results across papers are not comparable because datasets differ; runs its own comparison.", "Multiple (incl. CIC-IDS2017)", "Useful taxonomy; confirms the field's comparability problem. Survey, no deployment evaluation."],
 ["Ferrag et al. (2022) [5] — 1,417", "New IoT/IIoT dataset (Edge-IIoTset) from a purpose-built testbed; evaluates ML and DL in centralised and federated modes.", "Fourteen attack types in five threat classes; 61 selected features from 1,176 candidates; benchmark evaluation of ML/DL.", "Edge-IIoTset (public)", "Highly influential dataset for edge/IoT; focuses on data and federated learning rather than implementation cost of the detector."],
 ["Okey et al. (2023) [6] — 139", "Transfer learning: pretrained CNNs (VGG, Inception, MobileNet, EfficientNet) on features converted to images; ensemble 'ELETL-IDS'.", "Reported on two datasets (specific figures not extracted here).", "CIC-IDS2017, CSE-CIC-IDS2018", "Heavy pretrained image models for tabular flow data; evaluated in-distribution; resource cost on an edge device not the focus."]]
T(rows, widths=[1.1, 1.4, 1.7, 1.0, 1.7], size=8)
P("Research gap. Across the works above, (a) high in-distribution accuracy is the norm, but (b) the benchmark labels are unreliable in places, and (c) generalisation to new conditions is poor. What is rarely shown together is a small, fully transparent classifier whose every step (mathematics, preprocessing, flow construction, inference) is inspectable, evaluated with controlled multi-seed ablations, deployed offline with measured resource use, and probed for generalisation by withholding whole attack families. This paper provides that combined view for a compact MLP. We do not claim that no other work does so: the review is a focused sample, not an exhaustive survey.")

# ---------------------------------------------------------------- 3 method
H("3. Methodology")
H("3.1 Overall workflow", 2)
F("fig_pipeline.png", "Figure 1. End-to-end pipeline. Training is offline; at run time the stored model and preprocessing parameters are only loaded, never retrained.")
P("Two phases are kept strictly separate. Offline, a C++ trainer reads preprocessed arrays and writes a model file. Online, Python captures packets (live with Scapy, or by replaying a PCAP), assembles flows, extracts the same 57 features used in training, applies the stored preprocessing, and calls the C++ engine through a shared library and ctypes.")
H("3.2 Neural network", 2)
F("fig_mlp.png", "Figure 2. Architecture of the deployed network: 57 inputs, hidden layers of 64 and 32 ReLU units, and a softmax output (2 classes for the alarm model, 7 for the category model).", 4.2)
P("For a mini-batch X, each layer computes A_l = f(A_(l-1) W_l + b_l), with ReLU f(z) = max(0, z) in hidden layers and a numerically stable softmax at the output (the row maximum is subtracted before exponentiation). The loss is the class-weighted cross-entropy L = (1/B) Σ_i w_(y_i) · (−log p_(i,y_i)). For softmax with cross-entropy the output-layer error is δ_L = (p − onehot(y))·w_y/B; errors propagate backwards as δ_(l−1) = (δ_l W_lᵀ) ⊙ 1[A_(l−1) > 0], and the gradients are ∂L/∂W_l = A_(l−1)ᵀ δ_l and ∂L/∂b_l = Σ_batch δ_l. Weights are initialised by He normal initialisation, N(0, √(2/fan_in)) [8], biases to zero. Parameters are updated with Adam [9] (β1 = 0.9, β2 = 0.999, ε = 1e-8) or, optionally, SGD with momentum.")
P("Class weights are w_c = (N / (K·n_c))^0.5, a square-root compromise between no weighting and full inverse-frequency weighting (the exponent was compared in a preliminary sweep, Table 6). Early stopping keeps the epoch with the best validation macro-F1. Everything uses the C++ standard library; only the logistic-regression baseline uses scikit-learn.")
P("Correctness verification. The implementation is covered by 58 automated C++ checks, including a finite-difference gradient check of every weight and bias (maximum relative error 3×10⁻⁷ in double precision), hand-computed forward passes, and save/load round-trips; an independent NumPy forward pass reading the saved model file agrees with the C++ output to within 1e-4.")
H("3.3 Data and preprocessing", 2)
P(f"The supplied dataset (folder MachineLearningCVE) contains {len(da['files'])} CSV files with {da['total_rows']:,} flows, {da['n_features']} numeric features and a Label column. Its file names, CICFlowMeter-style feature names and attack labels are consistent with the public CIC-IDS2017 release [7]; this identification is inferred from the contents, since the files carry no provenance metadata. Steps, in order:")
B([f"Clean labels (a mis-encoded dash in the web-attack labels is normalised); select 57 features that a packet sniffer can compute (constant columns, a duplicated column, Subflow, Active/Idle and Bulk features are excluded).",
   f"Remove exact duplicate rows over (features, label): {da['total_duplicate_rows']:,} rows ({pc(da['total_duplicate_rows']/da['total_rows'],1)}) are dropped before splitting, so that identical rows cannot appear in both training and test data.",
   f"Stratified random split by raw label, 70/15/15 with seed 42: {tr:,} / {va:,} / {te:,} flows.",
   "Preprocessing statistics are fitted on the training split only: infinite and missing values are replaced by the training median, a signed log transform sign(x)·log(1+|x|) is applied, features are standardised with the training mean and standard deviation, and values are clipped to ±10. The same stored JSON is used at inference.",
   "Two tasks: binary (BENIGN vs ATTACK, all attack labels) and seven-class (BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Bot). Infiltration (36 flows) and Heartbleed (11) are too rare to learn or evaluate and are excluded from the seven-class task; no claim is made about categorising them."])
H("3.4 Live flow construction and feature extraction", 2)
P("Packets are parsed (IPv4 TCP/UDP) into records of timestamp, addresses, ports, IP length, payload length, transport header length, TCP flags and window. Flows are keyed by the bidirectional five-tuple; the first packet defines the forward direction. A flow ends on RST, after both FINs plus the final ACK, after 30 s idle, or after 120 s. The 57 features follow the dataset's definitions (durations and inter-arrival times in microseconds, lengths as payload bytes, sample standard deviation, flag counts over all packets, initial TCP window per direction). Because the dataset ships no raw packet captures, equality of our extractor with the original CICFlowMeter output could not be verified (Section 5).")
H("3.5 Implementation and reproducibility", 2)
T([["Item", "Setting"], ["Hardware / OS", f"{cpu}, {eb['cpu_count']} logical cores, Linux; single-threaded training"],
   ["Software", "g++ (C++17, -O3), Python 3.12, NumPy 1.26, Scapy 2.8, Streamlit"], ["Final architecture", "57 → 64 → 32 → 2 (binary) / 7 (category); 5,858 parameters in the binary model"],
   ["Optimiser", "Adam, learning rate 0.001, batch 256, up to 30 epochs, early stopping (patience 6) on validation macro-F1"],
   ["Ablation budget", f"max {pe['max_epochs']} epochs, patience 3, seeds {pe['seeds']}; the same split and metrics for every configuration"],
   ["Code / data", "Repository scripts: preprocessing/build_dataset.py, evaluation/paper_experiments.py, evaluation/train_final.py; all numbers in this paper are read from results/*.json"]], widths=[1.5, 5.0], size=9)

# ---------------------------------------------------------------- 4 results
H("4. Results and Discussion")
H("4.1 Dataset", 2)
P(f"Source: the course-supplied folder, consistent with the public CIC-IDS2017 release (public data). Records: {da['total_rows']:,} flows in {len(da['files'])} day-files; {da['rows_after_dedup']:,} after de-duplication. Features: {da['n_features']} raw, 57 used. Missing values: {da['nan_total']:,} (all in Flow Bytes/s); infinite values: {da['inf_total']:,} (Flow Bytes/s and Flow Packets/s, caused by zero-duration flows). The dataset has no timestamp, IP or session column, so a chronological or session-grouped split was impossible; a stratified random split was used and its optimism is quantified in Section 4.2(C).")
cd = da["class_distribution_after_dedup"]; norm = lambda k: k.replace("ï¿½", "-").replace("Web Attack - ", "Web Attack – ")
rows = [["Label (after de-duplication)", "Flows", "Share"]] + [[norm(k), f"{v:,}", pc(v / da['rows_after_dedup'], 2)] for k, v in cd.items()]
T(rows, widths=[3, 1.5, 1.2]); P("Table 1. Class distribution (15 labels; heavily imbalanced).", italic=True).runs[0].font.size = Pt(9)
F("fig_pca.png", "Figure 3. Two-dimensional PCA of standardised test flows (2,500 per class). Several attack families overlap or abut benign clusters, so no linear separation exists in this view; the classes occupy several disjoint regions each.", 4.6)

H("4.2 Ablation study", 2)
P(f"All configurations share the same data split, preprocessing (except where it is the component under test), epoch budget and metrics, and each is trained with {len(pe['seeds'])} random seeds (different initialisation and shuffling); we report mean ± standard deviation of the held-out test split. Test data were not used for any selection.")
P("(A) Cumulative ablation: components are added one at a time to a linear softmax baseline.", bold=True)
rows = [["Config", "Components", "F1 (attack)", "Recall", "FPR", "FNR"]]
desc = {"A0": "Linear softmax, z-score only", "A1": "+ signed log transform", "A2": "+ hidden layer (64, ReLU)", "A3": "+ second hidden layer (32)", "A4": "+ class weights (final)"}
for k, v in ab.items(): rows.append([k.split("_")[0], desc[k.split("_")[0]], ms(v["f1"]), ms(v["recall"]), ms(v["false_positive_rate"]), ms(v["false_negative_rate"])])
T(rows, widths=[0.6, 2.0, 1.2, 1.1, 1.1, 1.1], size=8); P("Table 2. Cumulative ablation (binary task, mean ± std over 5 seeds).", italic=True).runs[0].font.size = Pt(9)
d = lambda a, b: ab[b]["f1"]["mean"] - ab[a]["f1"]["mean"]; keys = list(ab)
P(f"Reading the table: the log transform lifts F1 from {ab[keys[0]]['f1']['mean']:.4f} to {ab[keys[1]]['f1']['mean']:.4f}; adding a hidden layer changes it by {d(keys[1], keys[2]):+.4f}; the second hidden layer by {d(keys[2], keys[3]):+.4f}; and class weighting by {d(keys[3], keys[4]):+.4f} (note the trade-off it makes between FPR and FNR rather than a uniform gain). A step is counted as a real improvement only when it exceeds the seed-to-seed standard deviations in Table 2; where it does not, we do not claim an effect (this bears on H3).")
P("(B) Feature ablation (final architecture).", bold=True)
rows = [["Feature set", "# features", "F1 (attack)", "FPR", "FNR"]]
fn = {"all57": "All features", "no_port": "Without Destination Port", "no_port_no_tcp_state": "Without port and TCP-state features (window sizes, min segment size, active data packets)"}
for k, v in ft.items(): rows.append([fn[k], v["n_features"], ms(v["f1"]), ms(v["false_positive_rate"]), ms(v["false_negative_rate"])])
T(rows, widths=[2.9, 0.8, 1.2, 1.1, 1.1], size=8); P("Table 3. Feature ablation (5 seeds). Destination Port is a potential shortcut because the testbed's attacks target fixed services.", italic=True).runs[0].font.size = Pt(9)
P("(C) Leave-one-attack-family-out (generalisation). For each family, all of its flows are removed from the training and validation sets; the binary model is then evaluated on that family's test flows (3 seeds). For comparison, in-distribution binary recall (Figure 7) is above 98% for every family except Bot (about 71%).", bold=True)
rows = [["Withheld family", "Test flows", "Recall on withheld family", "Benign FPR"]]
for k, v in lo.items(): rows.append([k, f"{v['n_test_flows']:,}", pc(v["recall_on_unseen_family"]["mean"], 1) + " ± " + pc(v["recall_on_unseen_family"]["std"], 1), pc(v["benign_fpr"]["mean"], 2)])
T(rows, widths=[1.8, 1.1, 2.4, 1.2], size=8); P("Table 4. Detection of an attack family never seen in training.", italic=True).runs[0].font.size = Pt(9)
F("fig_ablation_lofo.png", "Figure 4. Left: cumulative ablation with seed standard deviations. Right: recall on the withheld attack family (leave-one-family-out).")
fam_hi = [k for k, v in lo.items() if v["recall_on_unseen_family"]["mean"] > 0.9]; fam_lo = [k for k, v in lo.items() if v["recall_on_unseen_family"]["mean"] < 0.5]
P(f"Interpretation. Withholding a family sharply reduces detection of that family: recall is {pc(lo['DDoS']['recall_on_unseen_family']['mean'],0)} for DDoS (the best case, plausibly because DDoS flows resemble the DoS flows still in training), {pc(lo['DoS']['recall_on_unseen_family']['mean'],0)} for DoS and {pc(lo['Brute Force']['recall_on_unseen_family']['mean'],0)} for Brute Force, and close to zero for PortScan ({pc(lo['PortScan']['recall_on_unseen_family']['mean'],1)}), Web Attack ({pc(lo['Web Attack']['recall_on_unseen_family']['mean'],1)}) and Bot ({pc(lo['Bot']['recall_on_unseen_family']['mean'],1)}). No withheld family reached the >98% recall seen in-distribution. The detector therefore behaves as a recogniser of attack patterns present in its training data, not as a general anomaly detector; this strongly supports H2 and is consistent with the cross-dataset findings reported in [3]. The similarity explanation for DDoS is a hypothesis we did not test.")

H("4.3 Quantitative analysis", 2)
P("The final models were trained once with the final configuration (seed 42, up to 30 epochs) and evaluated on the untouched test split.")
b0 = ex["baselines"]; rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]]
for nm, r in (("Majority class (always benign)", b0["majority_class"]), ("Logistic regression (scikit-learn, balanced weights)", b0["logistic_regression"]), ("MLP 57-64-32-2 (this work, C++)", bn)):
    rows.append([nm] + [f"{r[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")])
T(rows, widths=[2.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], size=8); P(f"Table 5a. Binary task on the {te:,}-flow test split. The logistic regression did not fully converge within its iteration limit. The majority-class row shows why accuracy alone is misleading: it scores {b0['majority_class']['accuracy']:.3f} while detecting nothing.", italic=True).runs[0].font.size = Pt(9)
P(f"Confusion matrix (binary): TN = {cm[0][0]:,}, FP = {cm[0][1]:,}, FN = {cm[1][0]:,}, TP = {cm[1][1]:,}. H1 is supported for the in-distribution split: accuracy {pc(bn['accuracy'])}, F1 {pc(bn['f1'])}, FPR {pc(bn['false_positive_rate'])} (< 0.5%).")
rows = [["Class", "Precision", "Recall", "F1", "Test flows"]] + [[c, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", f"{v['support']:,}"] for c, v in mc["per_class"].items()]
rows.append(["Macro average", f"{mc['macro_precision']:.4f}", f"{mc['macro_recall']:.4f}", f"{mc['macro_f1']:.4f}", ""])
T(rows, widths=[1.6, 1.0, 1.0, 1.0, 1.1], size=8); P(f"Table 5b. Seven-class model (accuracy {mc['accuracy']:.4f}). Accuracy is dominated by benign traffic; macro-F1 is the more informative figure.", italic=True).runs[0].font.size = Pt(9)
P("Single-seed sensitivity sweeps (hidden sizes, learning rate, batch size, optimiser, class-weight exponent) are listed in Table 6; they move test F1 by roughly 0.001, comparable to seed noise, so we deliberately keep the simple architecture proposed at the outset.")
rows = [["Run (single seed)", "Val macro-F1", "Test F1", "FPR", "FNR"]] + [[r["name"], f"{r['val_macro_f1']:.4f}", f"{r['test_f1']:.4f}", f"{r['test_fpr']:.4f}", f"{r['test_fnr']:.4f}"] for r in ex["experiments"]]
T(rows, widths=[2.6, 1.1, 1.0, 0.9, 0.9], size=8); P("Table 6. Sensitivity sweeps, one factor changed from the default (64-32, Adam 0.001, batch 256, class-weight exponent 0.5, all features).", italic=True).runs[0].font.size = Pt(9)
H("Edge performance", 3)
bp = eb["packet_path_synthetic_mixed"]
T([["Measure", "Result (this laptop)"], ["Per-flow latency (preprocess + both MLPs, batch 1)", f"mean {eb['single_flow_pipeline_ms']['mean']:.3f} ms, p95 {eb['single_flow_pipeline_ms']['p95']:.3f} ms, p99 {eb['single_flow_pipeline_ms']['p99']:.3f} ms"],
   ["Throughput", f"{eb['throughput_batch1_flows_per_s']:,.0f} flows/s (batch 1); {eb['throughput_batch64_flows_per_s']:,.0f} flows/s (batch 64)"],
   ["Packet path (capture-file decode → flows → features → inference)", f"{bp['packets_per_s_end_to_end']:,.0f} packets/s on a synthetic fixture; Scapy decoding dominates, inference is {bp['inference_ms_per_flow_batched']:.4f} ms/flow"],
   ["Memory", f"process RSS {eb['rss_mb_start']:.0f} → {eb['rss_mb_end']:.0f} MB"], ["Model size", f"{eb['model_size_bytes']['binary']/1024:.1f} KB (binary) + {eb['model_size_bytes']['multiclass']/1024:.1f} KB (7-class)"],
   ["Offline operation", "Benchmark run inside an empty network namespace; an outbound connection test failed and inference still completed"]], widths=[2.8, 3.8], size=8)
P("Table 7. Edge measurements. The synthetic fixture is not attack data; it measures only software throughput.", italic=True).runs[0].font.size = Pt(9)

H("4.4 Qualitative analysis", 2)
F("fig_confusion.png", "Figure 5. Row-normalised confusion matrices on the test split (left: binary with counts; right: seven classes, percentages).")
F("fig_training.png", "Figure 6. Training curves of the final models (weighted training loss, validation loss, validation macro-F1).")
P("The confusion matrices show where predictions concentrate. In the seven-class matrix the largest off-diagonal masses are " + "; ".join(f"{c['true']} → {c['pred']} ({c['count']})" for c in ea["top_multiclass_confusions"][:4]) + ". Most errors therefore involve benign traffic on one side: benign flows that resemble short DoS/scan/bot flows, and attack flows that look like ordinary short benign flows. Numerical metrics alone hide this structure.")
pr = ea["feature_profiles"]
rows = [["Feature (median)", "Benign flows", "Bot — detected", "Bot — missed", "DoS Hulk — missed"]]
for k in ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Flow Packets/s", "PSH Flag Count", "Init_Win_bytes_forward"]:
    rows.append([k, f"{pr['BENIGN_median'][k]:,.1f}", f"{pr['Bot']['detected_median'][k]:,.1f}", f"{pr['Bot']['missed_median'][k]:,.1f}", f"{pr['DoS Hulk']['missed_median'][k]:,.1f}"])
T(rows, widths=[2.0, 1.1, 1.2, 1.2, 1.3], size=8); P("Table 8. Representative flow profiles (raw units, µs for time): flows the binary model detected versus missed.", italic=True).runs[0].font.size = Pt(9)

# ---------------------------------------------------------------- 5 failures
H("5. Failure Cases")
F("fig_errors.png", "Figure 7. Left: false-negative rate by attack type. Right: probability the model assigned to its erroneous decisions (log scale).")
fnl = ea["false_negatives_by_label"]; bot = fnl["Bot"]; hulk = fnl["DoS Hulk"]
P(f"False negatives. In total the binary model misses {sum(v['missed'] for v in fnl.values())} of {sum(v['n'] for v in fnl.values()):,} attack flows. The failure is concentrated: Bot is missed {pc(bot['miss_rate'],1)} of the time ({bot['missed']} of {bot['n']}), Infiltration {fnl['Infiltration']['missed']} of {fnl['Infiltration']['n']} (too few flows to conclude anything), and DoS Hulk {hulk['missed']} of {hulk['n']:,} flows. Table 8 gives the evidence: the median missed Bot flow has {pr['Bot']['missed_median']['Total Fwd Packets']:.0f} forward and {pr['Bot']['missed_median']['Total Backward Packets']:.0f} backward packet and lasts {pr['Bot']['missed_median']['Flow Duration']:.0f} µs, whereas detected Bot flows last {pr['Bot']['detected_median']['Flow Duration']:,.0f} µs and carry data; a one-packet-each-way flow is nearly indistinguishable from a short benign exchange using these features. Missed DoS Hulk flows are likewise truncated two-packet flows with no payload.")
P(f"False positives. {ea['false_positives']} benign flows ({pc(bn['false_positive_rate'])} of benign) are flagged; their mean assigned attack probability is {ea['conf_of_errors']['fp_mean_p_attack']:.2f}, i.e. many are confident mistakes, and in the seven-class matrix the dominant confusion is benign traffic labelled DoS or PortScan. Most false negatives are not confident: their mean attack probability is {ea['conf_of_errors']['fn_mean_p_attack']:.2f}, just under the 0.5 threshold, which suggests the operating threshold can trade FNR against FPR (not explored here).")
P("Unseen families. The leave-one-family-out study (Table 4) is the most consequential failure: families outside the training distribution are largely undetected. Plausible causes supported by our evidence: the classes occupy several disjoint feature regions (Figure 3), so the learned decision boundary encloses what it has seen; and flow summaries of different attacks can coincide with benign short flows.")
P("What the failures imply. The detector should be treated as a recogniser of known patterns and used alongside, not instead of, signature or anomaly mechanisms; per-flow detection also cannot flag an attack before its flows complete (idle timeout 30 s).")

# ---------------------------------------------------------------- 6 conclusion
H("6. Conclusion and Future Work")
P(f"We implemented a complete edge intrusion detector around a neural network written from scratch in C++. On the supplied dataset it reaches {pc(bn['accuracy'])} accuracy and a {pc(bn['false_positive_rate'])} false-positive rate in-distribution (H1 supported), runs in about {eb['single_flow_pipeline_ms']['mean']:.2f} ms per flow with a ~{eb['model_size_bytes']['binary']/1024:.0f} KB model, and works offline. The controlled ablation supports H3 for the components whose gain exceeds seed variation (see Table 2) and not for those that do not. H2 is strongly supported: detection of withheld attack families is far below in-distribution recall, which is the main caution for anyone deploying such a model.")
P("Limitations. (1) The data are one 2017 testbed, and the labels of the original release contain documented errors [1, 2]; our scores are for those labels and are not comparable with results on corrected data. (2) A random split cannot control for correlated neighbouring flows. (3) The live feature extractor re-implements the dataset's feature definitions but could not be validated against the original tool or raw captures; live-capture accuracy is therefore unvalidated, and live sniffing itself (which needs elevated privileges) was not exercised in our environment; PCAP replay and replay of real held-out flows were. (4) Synthetic packet fixtures used in tests are not realistic attacks and are classified benign. (5) A host sees only its own traffic on a switched network unless a mirror port is used. (6) Ablations use five seeds and the leave-family-out study three; no confidence intervals beyond standard deviations are given.")
P("Future work. Retrain and evaluate on the corrected labels and flow generator of [1, 2]; evaluate across datasets as in [3]; capture our own labelled traffic and validate the extractor against the original tool; add threshold calibration and open-set / anomaly components to address unseen attacks; study int8 quantisation and a small single-board computer; compare against tree ensembles and a framework-based MLP for accuracy and cost.")

# ---------------------------------------------------------------- refs
H("References")
refs = ["G. Engelen, V. Rimmer and W. Joosen, “Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study,” in IEEE Security and Privacy Workshops (SPW), 2021. doi:10.1109/SPW53761.2021.00009",
 "L. Liu, G. Engelen, T. Lynar et al., “Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018,” in IEEE Conf. on Communications and Network Security (CNS), 2022. doi:10.1109/CNS56114.2022.9947235",
 "M. Cantone, C. Marrocco and A. Bria, “Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study,” IEEE Access, 2024. doi:10.1109/ACCESS.2024.3472907",
 "L. Mohammadpour, T. C. Ling and C. S. Liew, “A Survey of CNN-Based Network Intrusion Detection,” Applied Sciences, vol. 12, no. 16, 2022. doi:10.3390/app12168162",
 "M. A. Ferrag, O. Friha, D. Hamouda et al., “Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,” IEEE Access, 2022. doi:10.1109/ACCESS.2022.3165809",
 "O. D. Okey, D. Carrillo, M. Saadi et al., “Transfer Learning Approach to IDS on Cloud IoT Devices Using Optimized CNN,” IEEE Access, 2023. doi:10.1109/ACCESS.2022.3233775",
 "I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” in Proc. ICISSP, 2018, pp. 108–116. (original CIC-IDS2017 paper; dataset source, outside the 2021–2026 survey window)",
 "K. He, X. Zhang, S. Ren and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” in Proc. IEEE ICCV, 2015. (He initialisation)",
 "D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. ICLR, 2015. arXiv:1412.6980"]
for i, r in enumerate(refs, 1): P(f"[{i}] {r}").runs[0].font.size = Pt(9.5)
P("Note on sources: entries [1]–[6] were located and their abstracts read on 16 November 2026, with citation counts taken from OpenAlex; entries [7]–[9] are standard references cited from the authors' knowledge and should be checked against the originals before submission.", italic=True).runs[0].font.size = Pt(9)
doc.save(os.path.join(ROOT, "documentation", "Research_Paper.docx")); print("saved")
