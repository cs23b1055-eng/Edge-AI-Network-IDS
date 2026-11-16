"""Builds documentation/Presentation.pptx (12 slides, required order). Numbers come from results/*.json."""
import json, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
J = lambda n: json.load(open(os.path.join(ROOT, "results", n)))
da, sp, fe, ex, pe, eb, ea = J("dataset_analysis.json"), J("split_report.json"), J("final_evaluation.json"), J("experiments.json"), J("paper_experiments.json"), J("edge_benchmark.json"), J("error_analysis.json")
FIG = os.path.join(ROOT, "documentation", "figures"); bn, mc = fe["binary"], fe["multiclass"]; ab, ft, lo = pe["ablation"], pe["features"], pe["leave_one_family_out"]
pc = lambda x, d=2: f"{x*100:.{d}f}%"
NAVY, BLUE, GREY, ORANGE = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x3B, 0x5B, 0x8C), RGBColor(0x44, 0x44, 0x44), RGBColor(0xC9, 0x76, 0x3B)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

def slide(title, num, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6]); bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.0)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(11.5), Inches(0.7)); p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = title; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
    nb = s.shapes.add_textbox(Inches(12.2), Inches(7.0), Inches(1), Inches(0.4)); q = nb.text_frame.paragraphs[0]; q.alignment = PP_ALIGN.RIGHT; rr = q.add_run(); rr.text = f"{num}/12"; rr.font.size = Pt(11); rr.font.color.rgb = GREY
    s.notes_slide.notes_text_frame.text = notes; return s
def bullets(s, items, x=0.5, y=1.25, w=6.2, h=5.5, size=18):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        sub = it.startswith("  "); p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = p.add_run(); r.text = ("– " if sub else "• ") + it.strip()
        r.font.size = Pt(size - 3 if sub else size); r.font.color.rgb = GREY if sub else RGBColor(0x22, 0x22, 0x22); p.space_after = Pt(6); p.level = 1 if sub else 0
def pic(s, name, x, y, w=None, h=None):
    s.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)
def table(s, rows, x, y, w, h, size=12, colw=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j); cell.text = str(c) or " "; para = cell.text_frame.paragraphs[0]; para.runs[0].font.size = Pt(size); para.runs[0].font.bold = (i == 0)
            if i == 0: cell.fill.solid(); cell.fill.fore_color.rgb = BLUE; para.runs[0].font.color.rgb = RGBColor(255, 255, 255)
    if colw:
        for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)
def caption(s, t, x, y, w): 
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.5)).text_frame; tf.word_wrap = True; r = tf.paragraphs[0].add_run(); r.text = t; r.font.size = Pt(11); r.font.italic = True; r.font.color.rgb = GREY

# 1 Introduction
s = slide("Edge AI Network Intrusion Detection with a From-Scratch Neural Network", 1, "Say: networks are attacked all the time. A detector that runs on the device itself is fast, private and works offline. I built the whole thing, including the neural network maths in C++, and then tested honestly how far the high accuracy numbers can be trusted. Team names go on this slide.")
bullets(s, ["Problem: detect malicious network flows (DoS, DDoS, port scan, brute force, web attacks, bots)", "Motivation: detection on the edge device – low latency, privacy, works offline",
 "Gap: papers report >99% accuracy, but benchmark labels have errors and models degrade on new data",
 "Objectives:", "  Build an MLP from scratch in C++ and deploy it end-to-end on a laptop", "  Measure it with controlled ablations and test generalisation honestly"], w=12.3, size=22)
caption(s, "[Team members · Course DIP / CN / COMA · Institute]", 0.5, 6.6, 9)
# 2 Literature
s = slide("Literature Survey (2021–2026)", 2, "Six verified papers; citation counts all from OpenAlex on 16 Nov 2026. Key message: (1) Engelen 2021 and Liu 2022 found errors in CIC-IDS2017; (2) Cantone 2024 found cross-dataset accuracy near chance; so in-distribution scores are not enough. Our gap: a small transparent model, controlled multi-seed ablations, offline deployment, and a generalisation test by withholding whole attack families.")
table(s, [["Paper (cites)", "What it shows"], ["Engelen et al. 2021 (304)", "CIC-IDS2017 has flow/feature/label errors; >20% of traces relabelled or rebuilt"],
 ["Liu, Engelen et al. 2022 (164)", "Many undocumented errors in CIC-IDS-2017/2018; corrected dataset released"], ["Cantone et al. 2024 (155)", "Near-perfect in-dataset, ~chance across datasets"],
 ["Mohammadpour et al. 2022 (253)", "CNN-IDS survey; results across papers not comparable"], ["Ferrag et al. 2022 (1,417)", "Edge-IIoTset: IoT/IIoT dataset, 14 attacks, centralised + federated ML"],
 ["Okey et al. 2023 (139)", "Transfer-learning CNN ensemble on CIC-IDS2017/2018"]], 0.5, 1.2, 12.3, 3.6, 13, [3.6, 8.7])
bullets(s, ["Research gap: transparent small model + controlled multi-seed ablation + offline edge deployment + leave-one-family-out generalisation test"], y=5.5, w=12.3, h=1.4, size=18)
# 3 Hypothesis
s = slide("Research Hypotheses", 3, "Three testable statements. H1 is about feasibility, H2 about generalisation, H3 about which components matter. At the end I will tell you which were supported by measurement.")
bullets(s, ["H1 – Feasibility: a from-scratch C++ MLP reaches >99% accuracy/F1 and FPR <0.5% on held-out data, <1 ms per flow, fully offline",
 "H2 – Generalisation: if an attack family is withheld from training, recall on it falls far below the in-distribution recall",
 "H3 – Components: log transform, a hidden layer and class weighting each improve F1 beyond seed-to-seed variation"], w=12.3, size=24)
# 4 Methodology
s = slide("Methodology", 4, "Walk the pipeline left to right. Training is offline and happens once. At run time Python only loads the model through ctypes and predicts. Preprocessing statistics come from the training split only, so no leakage.")
pic(s, "fig_pipeline.png", 0.4, 1.2, w=12.5)
bullets(s, ["Offline: C++ trainer → model file; Online: Python captures packets, builds flows, extracts 57 features, calls C++ via ctypes",
 "Preprocessing (train-only fit): NaN/inf → median, signed log1p, z-score, clip ±10", "Flows: 5-tuple, end on RST / FIN-FIN-ACK / 30 s idle / 120 s max"], y=3.9, w=12.3, h=3, size=19)
# 5 Architecture
s = slide("Model Architecture", 5, "Input 57 numbers, two hidden ReLU layers of 64 and 32 neurons, softmax output. 5,858 parameters, 23 KB. Forward: matrix multiply plus bias plus ReLU. Backward: chain rule; for softmax with cross-entropy the output error is just p minus the one-hot label. Adam updates. I verified backprop with a finite-difference gradient check: error 3e-7.")
pic(s, "fig_mlp.png", 0.4, 1.3, w=6.3)
bullets(s, ["57 → 64 (ReLU) → 32 (ReLU) → 2 / 7 (softmax); 5,858 parameters", "He initialisation, weighted cross-entropy, Adam (lr 0.001, batch 256)", "Backprop written by hand; verified by numerical gradient check (max rel. error 3×10⁻⁷)",
 "58 C++ unit checks; C++ output = independent NumPy forward (1e-4)", "Python ↔ C++ through shared library + ctypes"], x=6.9, y=1.3, w=6.1, size=17)
# 6 Dataset
s = slide("Dataset", 6, "One folder of 8 CSVs; it matches the public CIC-IDS2017 release in contents (inferred – no metadata). 2.83 million flows, 15 labels, 80% benign. I removed 308 thousand duplicates before splitting. Be ready to say: the original labels have documented errors, so our numbers are for the original labels.")
cd = da["class_distribution_after_dedup"]; top = list(cd.items())
bullets(s, [f"Source: supplied folder MachineLearningCVE (public, CIC-IDS2017-style; identification inferred)", f"{da['total_rows']:,} flows · {len(da['files'])} files · {da['n_features']} features (57 used) · 15 labels",
 f"Duplicates removed: {da['total_duplicate_rows']:,} → {da['rows_after_dedup']:,} flows", f"NaN: {da['nan_total']:,} · inf: {da['inf_total']:,} (zero-duration flows)",
 "No timestamps/IPs → stratified random split", "Known label errors in original release [Engelen 2021]"], w=6.4, size=16)
table(s, [["Label", "Flows"]] + [[k.replace("ï¿½", "–"), f"{v:,}"] for k, v in top[:9]] + [["… 6 more labels", ""]], 7.2, 1.3, 5.6, 4.6, 12, [3.9, 1.7])
# 7 Setup
s = slide("Experimental Setup", 7, "Same split, same budget for every ablation configuration; five seeds so that we can say what is real and what is noise. The test split was never used to choose anything.")
tr, va, te = (sp["splits"][k]["rows"] for k in ("train", "val", "test"))
bullets(s, [f"Split (stratified, seed 42): train {tr:,} · val {va:,} · test {te:,}", "Scaling statistics fitted on train only; duplicates removed before splitting",
 "Binary (benign vs attack) and 7-class (BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Bot)", "Infiltration (36) and Heartbleed (11) too rare → binary only",
 "Final training: Adam, lr 0.001, batch 256, ≤30 epochs, early stopping on val macro-F1", f"Ablation: ≤{pe['max_epochs']} epochs, seeds {pe['seeds']}, mean ± std",
 "Metrics: accuracy, precision, recall, F1, FPR, FNR, confusion matrix, per-class", "Baselines: majority class, logistic regression", "Hardware: 12-core laptop CPU, single thread, no GPU"], w=12.3, size=19)
# 8 Quantitative
b0 = ex["baselines"]; s = slide("Quantitative Results", 8, f"Binary: accuracy {pc(bn['accuracy'])}, recall {pc(bn['recall'])}, FPR {pc(bn['false_positive_rate'])}. Logistic regression has FPR {pc(b0['logistic_regression']['false_positive_rate'])}, about 18 times worse. Majority class gets 83% accuracy but detects nothing, which is why accuracy alone is misleading. Seven-class macro-F1 0.938; Bot is the weak class. Speed 0.05 ms per flow, ~145k flows/s, ~80 MB RAM, 23 KB model, runs offline.")
rows = [["Model", "Acc", "Prec", "Recall", "F1", "FPR", "FNR"]]
for nm, r in (("Majority class", b0["majority_class"]), ("Logistic regression", b0["logistic_regression"]), ("MLP (ours, C++)", bn)): rows.append([nm] + [f"{r[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")])
table(s, rows, 0.5, 1.25, 7.4, 1.7, 13, [2.3, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
rows = [["7-class", "P", "R", "F1"]] + [[c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}"] for c, v in mc["per_class"].items()]
table(s, rows, 0.5, 3.2, 7.4, 3.4, 12, [2.6, 1.6, 1.6, 1.6]); 
bullets(s, [f"Test set: {te:,} flows", f"7-class macro-F1 {mc['macro_f1']:.3f}", f"Latency {eb['single_flow_pipeline_ms']['mean']:.3f} ms/flow", f"{eb['throughput_batch64_flows_per_s']:,.0f} flows/s (batch)", f"RAM ≈ {eb['rss_mb_end']:.0f} MB · model {eb['model_size_bytes']['binary']/1024:.0f} KB", "Verified with no network access"], x=8.3, y=1.3, w=4.8, size=17)
# 9 Qualitative
s = slide("Qualitative Results", 9, "Left: confusion matrices – the mistakes almost always involve benign traffic on one side. Right: PCA of the features – attack classes sit in several separate regions and some touch benign clusters, so a model only knows what it has seen. In the dashboard (live demo) you can see alerts with category, confidence and the ground-truth label.")
pic(s, "fig_confusion.png", 0.3, 1.2, w=7.6); pic(s, "fig_pca.png", 8.0, 1.2, w=5.1)
caption(s, "Confusion matrices (test split) and PCA of test flows by class.", 0.5, 4.6, 7); 
bullets(s, ["Largest confusions: " + "; ".join(f"{c['true']}→{c['pred']} ({c['count']})" for c in ea["top_multiclass_confusions"][:3]), "Dashboard (live demo): alerts with category, confidence, timestamp, ground truth"], x=0.5, y=5.1, w=12.3, h=1.8, size=17)
# 10 Ablation & failures
s = slide("Ablation & Failure Cases", 10, "Ablation: add one component at a time with five seeds; bars show mean and standard deviation. Right: leave-one-family-out, i.e. recall for an attack family never seen in training. Failures: Bot flows are often single-packet-each-way flows that look like normal short flows. Say the generalisation result yourself - it is the most important finding.")
pic(s, "fig_ablation_lofo.png", 0.3, 1.15, w=7.3); pic(s, "fig_errors.png", 0.3, 4.15, w=7.0)
fnl = ea["false_negatives_by_label"]; lofo = "; ".join(f"{k} {pc(v['recall_on_unseen_family']['mean'], 0)}" for k, v in lo.items())
bullets(s, [f"Ablation F1: " + " → ".join(f"{ab[k]["f1"]["mean"]:.4f}" for k in ab), "(linear → +log1p → +hidden64 → +hidden32 → +class weights)",
 f"Unseen-family recall: {lofo}", f"Bot missed {pc(fnl['Bot']['miss_rate'],0)}; missed Bot flows: 1 pkt each way, ~40 µs", f"{ea['false_positives']} false alarms; DoS Hulk: {fnl['DoS Hulk']['missed']} misses (truncated 2-packet flows)"], x=7.7, y=1.3, w=5.4, size=15)
# 11 Discussion
s = slide("Discussion", 11, "Interpretation. High in-distribution scores come from a random split on one testbed. Recognising known patterns works; recognising unknown ones does not. The honest message for deployment: use it with other defences, threshold tuning and retraining on corrected, diverse data.")
sig = [k for k in list(ab)[1:] if ab[k]["f1"]["mean"] - ab[list(ab)[list(ab).index(k) - 1]]["f1"]["mean"] > ab[k]["f1"]["std"] + ab[list(ab)[list(ab).index(k) - 1]]["f1"]["std"]]
bullets(s, ["In-distribution: H1 supported (≈99.8% accuracy, FPR 0.12%, 0.05 ms/flow, offline)", "H2 supported: unseen families are largely missed → the model recognises known patterns, not 'attacks' in general",
 f"H3: gains beyond seed noise for: {', '.join(k.split('_', 1)[1] for k in sig) or 'none'}; other steps are within noise – we do not claim them",
 "Destination Port is a possible shortcut: removing it costs only a little F1 (see paper Table 3)", "Dataset caveat: original labels contain known errors → compare with corrected data before drawing real-world conclusions",
 "Live feature extractor re-implemented but not validated against the original tool; live sniffing needs root (not exercised here)"], w=12.3, size=19)
# 12 Conclusion
s = slide("Conclusion & Future Work", 12, "Contributions: from-scratch C++ MLP with verified backprop; packet-to-alert edge pipeline; controlled ablation and generalisation study; honest failure analysis. Future work as listed. Close by restating the key caution: do not trust in-distribution accuracy alone.")
bullets(s, ["Contributions", "  From-scratch C++ MLP (gradient-checked), ctypes bridge, offline edge pipeline + dashboard", "  Five-seed ablation, feature ablation, leave-one-family-out study, evidence-based error analysis",
 "Limitations", "  One 2017 testbed with label errors; random split; extractor unvalidated vs original tool; live capture untested here",
 "Future work", "  Corrected labels [Engelen, Liu] and cross-dataset evaluation [Cantone]", "  Own captured traffic; threshold calibration; open-set/anomaly detection for unseen attacks", "  int8 quantisation, single-board computer, comparison with tree ensembles"], w=12.3, size=20)
prs.save(os.path.join(ROOT, "documentation", "Presentation.pptx")); print("saved", len(prs.slides.__iter__.__self__._sldIdLst))
