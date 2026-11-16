# Edge AI Network Intrusion Detection System (MLP from scratch in C++)

A neural network (MLP) written from scratch in C++ (no ML framework) classifies network flows as benign/attack
(and attack category). Python captures packets, builds flows, extracts features and calls the C++ engine via
`ctypes`. Everything runs locally; a Streamlit dashboard shows live results. Full results: `documentation/FINAL_REPORT.md`.

## Dataset
Not included (too large for GitHub). Place the CSV files (CIC-IDS2017-style "MachineLearningCVE" folder) in a folder named
`MachineLearningCVE/` next to this project folder, then run `./run.sh data`. The trained models in `models/` already work without it.

Course project - B.Tech, Computer Networks (CN).

## Install (once)
```bash
cd AI_Network_IDS
./run.sh setup        # venv + python packages (needs internet only here)
./run.sh build        # compiles C++ trainer, shared library, C++ tests (needs g++)
```
Trained models are already in `models/`, so you can skip straight to "Run".

## Run
| Goal | Command |
|---|---|
| Dashboard (main demo) | `./run.sh dashboard` then open http://127.0.0.1:8501, choose a source, press **Start** |
| Detect real attack flows, terminal | `./run.sh demo 5000` |
| Analyse a PCAP file | `./run.sh pcap datasets/pcaps/synthetic_mixed.pcap` |
| Live capture (needs root/CAP_NET_RAW) | `./run.sh live eth0` |
| All tests | `./run.sh test` |
| Edge benchmark (offline namespace) | `./run.sh benchmark` |

Rebuild from scratch: `./run.sh data` (analysis + split + preprocessing), `./run.sh experiments` (~25 min), `./run.sh train`.

## Layout
`cpp_engine/` matrix, ReLU/softmax, MLP, backprop, Adam/SGD, trainer, C API, unit tests ·
`python_network/` packet_capture, flow_manager, feature_extractor, preprocessing, network_monitor ·
`integration/engine.py` ctypes bridge · `dashboard/app.py` · `preprocessing/` dataset scripts ·
`evaluation/` experiments, final training, benchmark · `models/` trained model + preprocessing.json ·
`results/` all measured outputs · `documentation/` reports · `tests/`.
The original dataset (`../MachineLearningCVE`) is only ever read.

## Important limitations
* Dataset flows come from CICFlowMeter; the live extractor re-implements those definitions but could not be verified
  against the original PCAPs (none provided). Treat live-capture accuracy as unvalidated.
* Synthetic PCAPs in `datasets/pcaps` test the plumbing only; they are *not* realistic attacks and the model labels them benign.
* A laptop only sees its own traffic (or what a mirror/monitor port provides) on a switched network.
