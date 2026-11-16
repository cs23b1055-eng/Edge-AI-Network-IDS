# Final Engineering Report

Every number below was measured by the scripts in this repository (see `results/`).

## 1. Implemented components

- **C++ MLP from scratch** (`cpp_engine/`): row-major matrix kernels, He init, ReLU, stable softmax, weighted cross-entropy, forward pass, backpropagation, mini-batch SGD+momentum and Adam, class weighting, early stopping on validation macro-F1, binary model save/load, C API.
- **Preprocessing** (`python_network/preprocessing.py`): NaN/inf -> train median, signed log1p, z-score (train-only fit), clip +-10; same class used for training and live inference; saved as `models/preprocessing.json`.
- **Networking**: Scapy PCAP/live capture (TCP/UDP, IPs, ports, timestamps, sizes, flags, windows), bidirectional 5-tuple flow assembly with FIN/RST/idle/active timeouts, 57-feature extractor.
- **Integration**: `ctypes` bridge with input validation (shape, NaN/inf), error codes and exceptions.
- **Edge monitor + Streamlit dashboard** (status, packets, flows, normal/suspicious, categories, confidence, alerts with timestamps, latency, CPU, RAM, model metrics, experiments).

## 2. Architecture

`packets -> flows -> 57 features -> Preprocessor -> [C++ binary MLP] + [C++ 7-class MLP] -> alert + live_state.json -> dashboard`

## 3. Dataset

`MachineLearningCVE/` (8 CSVs, 2,830,743 rows, 78 features + Label, 15 labels; CIC-IDS2017-style, inferred from content). 2,522,362 rows after removing 308,381 exact duplicates. No timestamps/IPs, so a stratified random 70/15/15 split (seed 42) was used. Details: `DATASET_REPORT.md`.

## 4. Neural network and training

57 inputs -> 64 (ReLU) -> 32 (ReLU) -> 2 (binary) or 7 (category); softmax + cross-entropy; Adam lr=0.001, batch 256, class weights (N/(K*n_c))^0.5, up to 30 epochs with early stopping (patience 6; binary stopped after 16 epochs, multiclass 8). Parameter count: binary 57*64+64+64*32+32+32*2+2 = 5,858; model files are ~23 KB. Hyper-parameters were chosen by validation macro-F1, never on test.

## 5. Results (held-out test split, 374,929 flows)

**Binary model**: accuracy 0.9983, precision 0.9942, recall 0.9960, F1 0.9951, **FPR 0.0012**, **FNR 0.0040**. Confusion matrix [[TN 310,681, FP 371], [FN 253, TP 63,624]].

**7-class model**: accuracy 0.9975, macro-F1 0.9378 (accuracy is dominated by benign traffic).

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| BENIGN | 0.9994 | 0.9977 | 0.9985 | 311,052 |
| DoS | 0.9877 | 0.9971 | 0.9924 | 29,058 |
| DDoS | 0.9993 | 0.9994 | 0.9994 | 19,202 |
| PortScan | 0.9876 | 0.9988 | 0.9932 | 13,623 |
| Brute Force | 0.9594 | 0.9978 | 0.9782 | 1,373 |
| Web Attack | 0.8873 | 0.9813 | 0.9320 | 321 |
| Bot | 0.6569 | 0.6860 | 0.6711 | 293 |

Weakest class: **Bot** (F1 0.67) - only 1,953 samples and it resembles benign traffic. Web Attack precision is 0.89. Infiltration/Heartbleed are not categorised (too few rows).

### Baselines (binary test)

| Model | Acc | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| majority_class | 0.8296 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| logistic_regression | 0.9783 | 0.9038 | 0.9767 | 0.9388 | 0.0213 | 0.0233 |

(The logistic-regression baseline hit its iteration limit without fully converging.) The MLP clearly beats it: FPR 0.0012 vs 0.0213.

### Experiments (binary; one factor changed from 64-32 / Adam 0.001 / batch 256 / all 57 features; 12 epoch cap)

| Run | Features | Val macro-F1 | Test F1 | FPR | FNR |
|---|---|---|---|---|---|
| hidden_32 | 57 | 0.9963 | 0.9942 | 0.0012 | 0.0056 |
| hidden_64,32 | 57 | 0.9966 | 0.9948 | 0.0012 | 0.0046 |
| hidden_128,64 | 57 | 0.9968 | 0.9952 | 0.0014 | 0.0030 |
| hidden_128,64,32 | 57 | 0.9964 | 0.9946 | 0.0015 | 0.0036 |
| lr_0.0003 | 57 | 0.9966 | 0.9946 | 0.0011 | 0.0052 |
| lr_0.003 | 57 | 0.9967 | 0.9948 | 0.0010 | 0.0056 |
| lr_0.01 | 57 | 0.9964 | 0.9942 | 0.0011 | 0.0063 |
| batch_64 | 57 | 0.9964 | 0.9943 | 0.0014 | 0.0047 |
| batch_1024 | 57 | 0.9968 | 0.9949 | 0.0012 | 0.0047 |
| opt_sgd_momentum_lr0.05 | 57 | 0.9966 | 0.9949 | 0.0011 | 0.0048 |
| cw_power_0 | 57 | 0.9968 | 0.9944 | 0.0009 | 0.0069 |
| cw_power_1 | 57 | 0.9959 | 0.9940 | 0.0021 | 0.0021 |
| features_no_dest_port | 56 | 0.9954 | 0.9927 | 0.0018 | 0.0059 |
| features_no_port_no_tcp_state | 52 | 0.9945 | 0.9910 | 0.0020 | 0.0084 |

Findings: architecture, learning rate, batch size and optimizer change test F1 by at most ~0.001 (0.9942-0.9952) - within what I would expect from seed noise of a single seed (one run each, no confidence intervals). Class weight power trades FPR against FNR. Removing `Destination Port` costs ~0.002 F1; removing TCP-state features costs ~0.004.

### Generalisation to unseen attacks (important)

Trained without PortScan and DDoS, the binary model detected only **0.7% of PortScan** and **38.4% of DDoS** flows (seen families: DoS Hulk 100.0%, FTP-Patator 100.0%). The model recognises attack *patterns present in training*; the high headline scores come from a random split of one testbed and will not transfer automatically to other networks or novel attacks.

## 6. Edge performance (this laptop, 12 cores, no network: run inside an empty network namespace)

- Offline: connection test failed (`offline_during_run=True`) and inference still ran.
- Latency per flow (preprocess + both MLPs via ctypes, batch 1): mean 0.048 ms, p95 0.061 ms, p99 0.088 ms.
- Throughput: 20,739 flows/s (batch 1), 148,270 flows/s (batch 64).
- Packet path on the synthetic fixture: 5,359 packets/s end-to-end (dominated by Scapy decoding); feature extraction 0.125 ms/flow.
- RAM: process RSS 64 -> 81 MB. CPU: 3.1 CPU-seconds for the whole benchmark. Model size: 22.9 KB (binary) + 23.6 KB (7-class).

## 7. Testing (all executed)

- C++ (`make -C cpp_engine test`): 58 checks pass - matrix ops, activations, forward (hand-computed), loss, **numerical gradient check (max relative error 3e-7)**, learning on XOR (Adam and SGD), save/load incl. corrupt files. One XOR seed hit a dead-ReLU minimum; the test uses a seed that doesn't (documented in the test).
- Python (`pytest tests/test_python.py`): 15 pass - label cleaning, train-only preprocessing, JSON round-trip, deployed-preprocessing == training arrays, split disjointness, flow direction/counts/FIN/RST/timeouts, hand-verified feature values, PCAP flow counts, **C++ output == independent numpy forward pass**, input validation, prediction on real test data, monitor end-to-end, real-dataset replay, dashboard rendering via Streamlit AppTest.
- Bugs found and fixed during development: out-of-memory in the first analysis script; wrong FPR definition (caught because FPR==FNR in every run; verified against a hand-computed case and experiments re-run); label mojibake; scapy link-layer not registered; bogus extra flow after TCP teardown; pyarrow/NumPy ABI mismatch.

## 8. Known limitations

1. **Live feature fidelity is unverified**: no original PCAPs; CICFlowMeter details (e.g. Active/Idle, flow termination, header-length definition) are re-implemented from its documented behaviour. Active/Idle/Bulk/Subflow features were dropped for this reason.
2. **Live sniffing could not be tested here** (needs root; `sudo` unavailable). PCAP replay and dataset replay are tested; live mode shows a clean error without privileges.
3. Synthetic PCAP port-scan/flood fixtures are classified benign: they do not resemble the dataset's attack flows. They validate plumbing only.
4. Random split from one 2017 testbed; poor transfer to unseen attack families (see above); Bot class weak; no multi-seed variance.
5. A laptop NIC sees only traffic to/from itself (plus broadcast/multicast) on a switched network; whole-LAN monitoring needs a mirror (SPAN) port, a hub, or running on the gateway/AP. Encrypted payloads are fine (flow features only) but VPN/tunnel traffic is seen as one flow.
6. Detection is per completed flow (after FIN/RST or timeout), so alerts lag the attack by up to the idle timeout (30 s).

## 9. How to run / present

See `README.md`. Suggested 10-minute presentation: (1) problem + dataset report (`DATASET_REPORT.md`); (2) show `cpp_engine/src/mlp.cpp` forward/backward and run `./run.sh test` (gradient check); (3) `./run.sh dashboard` -> *Dataset replay* -> Start, show alerts/categories/latency/ground-truth agreement; (4) *Model performance* and *Experiments* tabs, state the unseen-attack limitation yourself; (5) `./run.sh benchmark` to show offline operation; (6) optionally `./run.sh live eth0` with root on your own machine while running `nmap` against it from another host.
