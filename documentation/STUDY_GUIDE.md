# Study Guide — Edge AI Network Intrusion Detection System

Read in order. Parts: 1 big picture · 2 concepts · 3 file-by-file · 4 how it all runs · 5 numbers to remember · 6 likely questions · 7 demo script · 8 prompt for ChatGPT.

---

## 1. The big picture (say this in 30 seconds)

"Network traffic is made of packets. Packets between two computers on the same ports form a **flow**. For each flow I calculate 57 numbers (duration, packet counts, sizes, timing, TCP flags...). A neural network that I **wrote myself in C++** (no TensorFlow/PyTorch) reads those 57 numbers and says: **normal or attack**, and which kind of attack. It was trained on 2.8 million real labelled flows. It runs fully **on my laptop** (Edge AI, no cloud), and a dashboard shows alerts live."

Pipeline:

```
packets -> flows -> 57 features -> normalise -> C++ neural network -> alert -> dashboard
(Scapy)   (flow_manager) (feature_extractor) (preprocessing) (libidsengine.so)  (Streamlit)
```

Two phases (never mix them):
- **Training (offline, once):** C++ program `ids_train` learns weights from the dataset -> saves `models/*.bin`.
- **Inference (online, every flow):** Python loads the saved model through `ctypes` and only predicts. It never retrains.

---

## 2. Concepts you must understand

**Flow** — all packets of one conversation, identified by the 5-tuple: source IP, destination IP, source port, destination port, protocol. "Forward" = direction of the first packet; "backward" = the reply.

**Features** — numbers describing a flow, e.g. `Flow Duration`, `Total Fwd Packets`, `SYN Flag Count`. A DoS flood has many tiny packets very fast; a port scan has 1–2 packets per flow to many ports. The network learns these patterns.

**Dataset** — 8 CSV files, 2,830,743 flows, 78 columns + `Label` (BENIGN, DoS Hulk, PortScan, DDoS, FTP-Patator, ...). It looks like CIC-IDS2017 (I inferred that from the content; the files don't say).

**Preprocessing** (and why):
1. Remove duplicate rows *before* splitting (otherwise the same row is in train and test = cheating/leakage).
2. NaN/infinity -> replaced by the training median.
3. `signed log1p` (squashes huge values like byte rates).
4. Standardise: (x − mean)/std, with mean/std computed from the **training set only**. Same numbers saved in `models/preprocessing.json` and used live. If live used different scaling, the model would be wrong.
5. Split 70% train / 15% validation / 15% test. Validation = pick settings; test = final honest score, touched only for reporting.

**Neural network (MLP)** — layers of "neurons". Mine: 57 inputs → 64 → 32 → 2 outputs (or 7).
- Each layer: `z = a·W + b` (matrix multiply + bias).
- **ReLU** on hidden layers: `max(0, z)` — adds non-linearity so it can learn curves, not just lines.
- **Softmax** on output: turns scores into probabilities that sum to 1.
- **Cross-entropy loss**: `−log(probability of the correct class)`. Small when confident and right.
- **Forward propagation**: input → layers → probabilities.
- **Backpropagation**: apply the chain rule backwards to get the gradient of the loss w.r.t. every weight. For softmax+cross-entropy the output error is simply `(p − one_hot(label))`.
- **Gradient descent / mini-batch**: `W ← W − lr·gradient`, using 256 samples at a time. I use **Adam** (an improved version that adapts the step per weight); SGD+momentum is also implemented.
- **He initialisation**: random starting weights with std `sqrt(2/fan_in)` so ReLU signals don't vanish/explode.
- **Class weights**: attacks are rare (benign is ~80%), so mistakes on rare classes are weighted more.
- **Early stopping**: keep the model from the epoch with best validation macro-F1.
- **Epoch** = one pass over all training data.

**Metrics**
- Accuracy — misleading here (always guessing "benign" gives 83%! that is my "majority" baseline).
- Precision — of the flows I called attacks, how many really were. Recall — of the real attacks, how many I caught.
- F1 — balance of the two. Macro-F1 — average over classes, so rare classes count equally.
- **FPR** (false positive rate) — benign flows wrongly flagged (annoying false alarms). **FNR** — attacks missed (dangerous).
- Confusion matrix — table of true vs predicted.

**Edge AI** — the AI runs on the local device (my laptop), not a cloud server. Benefits: low latency, privacy, works offline. I proved offline by running the benchmark inside an empty network namespace.

**Python ↔ C++** — C++ is compiled into a shared library `libidsengine.so` exposing plain C functions (`ids_load`, `ids_predict_proba`, `ids_free`). Python's `ctypes` calls them. Float arrays are passed by memory pointer; error codes are returned.

---

## 3. File-by-file map

Project root: `AI_Network_IDS/`. The original data is **outside**, in `../MachineLearningCVE/` and is only ever read.

### `cpp_engine/` — the neural network from scratch
| File | What it does |
|---|---|
| `include/matrix.hpp` | `Matrix` struct (rows, cols, vector of floats) and the maths kernels: `matmul` (A·B), `matmul_AtB` (Aᵀ·B, used for weight gradients), `matmul_ABt` (A·Bᵀ, used to push error to the previous layer), `add_row_bias`, `col_sums`. |
| `include/activations.hpp` | ReLU, ReLU backward (derivative), numerically stable softmax (subtracts the row max before `exp`). |
| `include/mlp.hpp` | Declares the `Mlp` class, `Layer`, `TrainConfig` (lr, batch, epochs, optimizer). The "table of contents" of the network. |
| `src/mlp.cpp` | **The heart.** `Mlp()` constructor = He init; `forward()`; `loss()`; `backward()` = backpropagation; `step()` = Adam/SGD update; `save()`/`load()` binary model file with magic `IDSMLP01`. |
| `src/train_main.cpp` | Command-line trainer `ids_train`: reads the binary dataset files, shuffles, mini-batch loop, computes validation macro-F1 each epoch, early stopping, saves best model. Prints JSON per epoch. |
| `src/c_api.cpp` | Flat C interface for Python: `ids_load`, `ids_predict_proba`, `ids_n_in`, `ids_n_out`, `ids_free`, `ids_last_error`. Catches C++ exceptions and returns error codes. |
| `tests/test_engine.cpp` | 58 checks: matrix results by hand, activations, He-init statistics, hand-computed forward pass, loss, **numerical gradient check** (compares backprop to finite differences; error 3e-7), learning XOR, save/load, corrupt-file rejection. Built with `double` precision. |
| `Makefile` | Builds `build/ids_train`, `build/libidsengine.so`, `build/test_engine`. |

`build/` holds the compiled outputs.

### `preprocessing/` — dataset scripts (run once)
| File | What it does |
|---|---|
| `analyze_dataset.py` | Streams the 8 CSVs, counts rows, NaN, inf, duplicates, classes -> `results/dataset_analysis.json`. |
| `build_dataset.py` | Cleans labels, picks 57 features, removes duplicates, stratified split, fits the preprocessor **on train only**, writes binary files into `datasets/processed/` and `models/preprocessing.json`. |
| `make_dataset_report.py` | Generates `documentation/DATASET_REPORT.md` from the JSON results. |

### `python_network/` — networking + edge application
| File | What it does |
|---|---|
| `packet_capture.py` | `PacketRecord` (one parsed packet). `parse_packet()` extracts IPs, ports, size, payload, flags, window from a Scapy packet (TCP/UDP only). `pcap_source()` replays a file; `live_source()` sniffs a network interface (needs root). |
| `flow_manager.py` | Groups packets into bidirectional flows by 5-tuple. Ends a flow on RST, on FIN-FIN + final ACK, on idle timeout (30 s) or max length (120 s). |
| `feature_extractor.py` | Turns a finished flow into the 57 numbers, in the exact order the model expects (durations in microseconds like the dataset). |
| `preprocessing.py` | The feature list `FEATURES`, label groups, and the `Preprocessor` class (fit / transform / save / load). Used by **both** training and live inference -> consistency. |
| `network_monitor.py` | The orchestrator. `EdgeDetector` loads both models + preprocessing. `Monitor` runs: packet -> flow -> features -> prediction -> alert, measures latency/CPU/RAM, writes `results/live_state.json` every 0.5 s. Modes: `--pcap`, `--iface`, `--dataset-replay`. |

### `integration/engine.py`
`NativeModel` — the `ctypes` wrapper. Loads the `.so`, validates input shape and NaN/inf, calls C++, returns probabilities. Raises `EngineError` on problems.

### `evaluation/` — training, experiments, measurements
| File | What it does |
|---|---|
| `common.py` | Helpers: read/write the binary data format, call the C++ trainer. |
| `metrics.py` | Accuracy, precision, recall, F1, FPR, FNR, per-class, confusion matrix (written by hand). |
| `run_experiments.py` | Baselines (majority, logistic regression) + 14 experiments (hidden layers, learning rates, batch sizes, optimizer, class weights, feature sets) + the unseen-attack test -> `results/experiments.json`. |
| `train_final.py` | Trains the two deployed models and writes `models/binary_model.bin`, `models/multiclass_model.bin`, `results/final_evaluation.json`. |
| `edge_benchmark.py` | Measures latency, throughput, CPU, RAM, model size and offline operation -> `results/edge_benchmark.json`. |

### `models/` — what is deployed
`binary_model.bin` (normal vs attack, ~23 KB), `multiclass_model.bin` (7 categories), `preprocessing.json` (feature names, median, mean, std).

### `dashboard/app.py`
Streamlit app. Sidebar: choose source (dataset replay / PCAP / live), Start/Stop (launches `network_monitor.py` as a subprocess). Tabs: Live monitoring (reads `live_state.json` every second), Model performance, Experiments.

### `tests/`
`test_python.py` — 15 tests. `make_test_pcaps.py` — builds the *synthetic* PCAP fixtures in `datasets/pcaps/` (only for testing the plumbing).

### `results/`, `documentation/`, root
- `results/*.json` — every measured number (nothing typed by hand).
- `documentation/` — `DATASET_REPORT.md`, `FINAL_REPORT.md`, this guide.
- `run.sh` — one-word commands; `requirements.txt`; `README.md`.

---

## 4. What happens when you click Start (dataset replay)

1. Dashboard launches `network_monitor.py --dataset-replay N`.
2. It samples N real **test** flows (30% attacks), keeping their true labels hidden from the model.
3. For each flow: `Preprocessor.transform` -> `NativeModel.predict_proba` (C++) for the binary and 7-class models.
4. If P(attack) ≥ 0.5 -> alert with category, confidence, timestamp; also compares with the true label.
5. Writes `live_state.json`; the dashboard re-reads it each second and redraws.

For PCAP/live the only difference is steps 2–3 start from packets: `packet_capture -> flow_manager -> feature_extractor`.

---

## 5. Numbers to remember

- Dataset: **2,830,743** flows -> **2,522,362** after removing duplicates; 8 files; 15 labels.
- Model: 57 -> 64 -> 32 -> 2; **5,858 parameters**; ~23 KB.
- Binary test results: accuracy **99.83%**, precision 99.42%, recall **99.60%**, **FPR 0.12%**, **FNR 0.40%**.
- Baseline logistic regression: FPR 2.1% (≈18× worse). Majority class: 83% accuracy but catches 0 attacks.
- 7-class: macro-F1 **0.938**; weakest = Bot (F1 0.67).
- Speed: ~**0.05 ms** per flow, ~**145,000 flows/s**, ~80 MB RAM.
- Tests: **58** C++ checks + **15** Python tests, gradient check error **3e-7**.
- Unseen attack test: PortScan detected **0.7%**, DDoS **38%** when never seen in training.

---

## 6. Questions you may be asked (with honest answers)

**Why C++ and not TensorFlow?** To show and understand the mathematics (forward, backprop, Adam); C++ is also fast and small for edge devices. The only library use is C++ standard library; scikit-learn is used only for the logistic-regression *baseline*.

**How do you know your backpropagation is correct?** Gradient check: nudge each weight by ±1e-6, measure the loss change, compare to the backprop gradient. Max relative error 3e-7.

**Why not just use accuracy?** 83% of the flows are benign, so a useless model gets 83%. I report precision, recall, F1, FPR, FNR and a confusion matrix.

**Why did you remove duplicates / fit scaling on train only?** To prevent data leakage — otherwise test results are inflated.

**Why random split, not by time?** The dataset has no timestamps/IPs, so sessions can't be grouped. I state this limitation and test generalisation separately with a held-out-attack experiment.

**99.8% looks too good. Is it real?** It is real *for this dataset and split*, but it is optimistic: neighbouring flows of one attack burst look alike, and one testbed from 2017 was used. My held-out experiment shows the model fails on attack types it never saw (PortScan 0.7%). So it detects *known patterns*, not unknown attacks.

**Does it work on live traffic?** The live code path exists and the packet->flow->feature->prediction chain is tested on PCAP files, but I could not test live sniffing (needs root) and I could not verify my feature extractor against the original dataset's tool because the dataset has no PCAPs. That is the main limitation.

**Can the laptop see all the network's traffic?** No. On a switched network it sees only its own traffic plus broadcast/multicast. To monitor everything you need a mirror/SPAN port or to run on the router/gateway.

**What is "Edge AI" here?** Capture, feature extraction, inference, alerting and dashboard all run locally, offline (proved in an empty network namespace).

**Why two models?** Binary model decides alert/no alert (best at that: FPR 0.12%). The 7-class model names the category. Infiltration (36 rows) and Heartbleed (11 rows) are too rare to categorise, so I don't claim to.

**What would you improve?** Multi-seed experiments with confidence intervals; leave-one-attack-out training; validate on PCAPs / another dataset; int8 quantisation; test on a Raspberry Pi.

**Is it a research project?** Applied research: clear question, real data, baseline, ablations, negative results, reproducibility. Not novel science (MLPs on CIC-IDS2017 are well studied).

**Mistakes you found?** Out-of-memory in my first script; a wrong FPR formula (noticed because FPR equalled FNR in every run; fixed and re-ran); a bogus extra flow after TCP teardown; a library version clash.

---

## 7. Demo script (5–7 minutes)

1. `cd AI_Network_IDS && ./run.sh test` — show "58 checks, 0 failed" and "15 passed". (Say: gradient check proves backprop.)
2. Open `cpp_engine/src/mlp.cpp`; scroll to `forward`, `backward`, `step`. Explain in two sentences each.
3. `./run.sh dashboard` -> http://127.0.0.1:8501 -> *Dataset replay* -> Start. Point at: counters, categories chart, alerts with confidence, latency, CPU/RAM, "agreement with ground truth".
4. *Model performance* tab: confusion matrix, per-class table. *Experiments* tab: baseline vs MLP.
5. Say the limitations yourself (unseen attacks, live capture unverified, Bot weak). This builds credibility.

Backup if the dashboard fails: `./run.sh demo 5000` prints the same results in the terminal.

---

## 8. Prompt to give ChatGPT (copy everything in the box)

```
You are my presentation coach and technical tutor. I am a college student presenting this project tomorrow.
Teach me step by step, then quiz me. Use simple language, analogies, and short sections. Ask me to explain things
back to you and correct me. Do not invent facts beyond what I give you; if something is unknown, say so.

PROJECT: "Edge AI-Based Network Intrusion Detection System Using a Neural Network Built From Scratch"

SUMMARY
- Dataset: 8 CSV files, 2,830,743 network flows, 78 flow features + label, 15 labels (CIC-IDS2017-style: BENIGN, DoS Hulk,
  DDoS, PortScan, FTP/SSH-Patator, Web attacks, Bot, Infiltration, Heartbleed). 308,381 duplicates removed -> 2,522,362 rows.
  No timestamps/IPs, so a stratified random 70/15/15 split was used.
- Preprocessing: 57 packet-computable features; NaN/inf -> train median; signed log1p; z-score using TRAIN statistics only; clip +-10.
- Model: MLP written from scratch in C++ (no ML framework): 57 -> 64 (ReLU) -> 32 (ReLU) -> 2 (binary) or 7 (category), softmax,
  weighted cross-entropy, He init, backpropagation, Adam (SGD+momentum also), mini-batch 256, lr 0.001, early stopping on val macro-F1.
  5,858 parameters, ~23 KB file.
- Python <-> C++: C++ built as shared library libidsengine.so, called from Python with ctypes (load model, predict probabilities).
- Networking (Python/Scapy): packet capture (live or PCAP), 5-tuple bidirectional flow assembly (FIN/RST/idle timeout), 57-feature extractor,
  network_monitor orchestrates everything, writes live state JSON; Streamlit dashboard shows counters, alerts, confidence, latency, CPU/RAM.
- Edge AI: everything runs locally and offline (benchmark ran in an empty network namespace).
RESULTS (test split, 374,929 flows)
- Binary: accuracy 99.83%, precision 99.42%, recall 99.60%, FPR 0.12%, FNR 0.40%. Logistic-regression baseline FPR 2.1%.
  Majority-class baseline has 83% accuracy but 0 recall.
- 7-class: accuracy 99.75%, macro-F1 0.938; Bot is weakest (F1 0.67).
- Speed: ~0.05 ms per flow, ~145k flows/s, ~80 MB RAM.
- Tests: 58 C++ checks (incl. numerical gradient check, error 3e-7) and 15 Python tests, all passed.
- Hyper-parameter experiments (hidden layers, lr, batch size, optimizer, class weights, feature subsets) differ by only ~0.001 F1.
LIMITATIONS (important, I must say them honestly)
- Model trained without PortScan+DDoS detected only 0.7% PortScan and 38% DDoS -> it does not generalise to unseen attack types.
- Random split on a single 2017 testbed makes scores optimistic.
- Live sniffing needs root and was not tested; the live feature extractor could not be verified against the original tool (no PCAPs in dataset).
- Synthetic PCAP fixtures are plumbing tests only.
- A laptop only sees its own traffic on a switched network (needs a SPAN/mirror port for more).

FILES (project folder AI_Network_IDS/)
cpp_engine/include: matrix.hpp (matrix maths), activations.hpp (ReLU, softmax), mlp.hpp (network class)
cpp_engine/src: mlp.cpp (init, forward, loss, backprop, optimiser, save/load), train_main.cpp (trainer CLI), c_api.cpp (C interface for Python)
cpp_engine/tests/test_engine.cpp (58 checks)
preprocessing/: analyze_dataset.py, build_dataset.py, make_dataset_report.py
python_network/: packet_capture.py, flow_manager.py, feature_extractor.py, preprocessing.py, network_monitor.py
integration/engine.py (ctypes bridge); evaluation/: common.py, metrics.py, run_experiments.py, train_final.py, edge_benchmark.py
dashboard/app.py (Streamlit); models/ (binary_model.bin, multiclass_model.bin, preprocessing.json); tests/test_python.py; run.sh

PLAN FOR YOU
1. Teach me in this order: networking basics (packet, flow, 5-tuple, TCP flags) -> features -> preprocessing and data leakage ->
   neural network maths (forward, softmax, cross-entropy, backprop, gradient descent, Adam, He init) -> metrics (precision, recall, F1,
   FPR, FNR, why accuracy misleads) -> edge AI -> ctypes bridge -> limitations. After each part ask me 2 questions.
2. Then act as a strict examiner: ask me 15 viva questions one at a time, grade each answer out of 10, and show a better answer.
3. Finally help me write a 7-minute spoken script with slide titles (10 slides max).
Start with part 1 now.
```

Tip: after ChatGPT teaches each part, paste in the matching section of this guide to check its answers against the real project.

---

## 9. Research paper and slides (submission files)

Files: `documentation/Research_Paper.docx` (+ `.pdf`), `documentation/Presentation.pptx` (+ `.pdf`, 12 slides with speaker notes). Rebuild with `python documentation/build_paper.py` and `build_slides.py`; every number is read from `results/*.json`.

Extra experiments run for the paper (`evaluation/paper_experiments.py`, 5 seeds, same split):
- **Cumulative ablation** F1: linear 0.8449 -> +log1p 0.9327 -> +hidden64 0.9947 -> +hidden32 0.9949 -> +class weights 0.9946. Real gains: log transform and the first hidden layer. The last two steps are inside seed noise (say so).
- **Leave-one-family-out** (family never seen in training) recall: DDoS 60%, DoS 33%, Brute Force 21%, Web Attack 1.7%, PortScan 0.5%, Bot 0%. (The earlier "PortScan 0.7% / DDoS 38%" number removed *both* families together; this one removes one at a time. Quote the new table.)
- **Failures:** Bot is missed 29% of the time (86 of 293); missed Bot flows are 1 packet each way, about 40 microseconds, so they look like short normal flows.

Literature (citation counts from OpenAlex, 16 Nov 2026): Engelen 2021 (304), Liu 2022 (164), Cantone 2024 (155), Mohammadpour 2022 (253), Ferrag 2022 (1,417), Okey 2023 (139). Be ready to explain the gap: in-distribution scores, dataset label errors, poor cross-dataset generalisation.
