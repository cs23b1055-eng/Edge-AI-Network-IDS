#!/usr/bin/env bash
# Usage: ./run.sh <command>      (see README.md)
set -e; cd "$(dirname "$0")"; PY=.venv/bin/python
case "$1" in
  setup)     python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt ;;
  build)     make -C cpp_engine ;;
  data)      $PY preprocessing/analyze_dataset.py && $PY preprocessing/build_dataset.py && $PY preprocessing/make_dataset_report.py ;;
  experiments) $PY evaluation/run_experiments.py ;;
  train)     $PY evaluation/train_final.py --hidden 64,32 --lr 0.001 --batch 256 --epochs 30 --features all57 ;;
  test)      make -C cpp_engine test && $PY -m pytest tests/test_python.py -q ;;
  fixtures)  $PY tests/make_test_pcaps.py ;;
  benchmark) unshare -rn $PY evaluation/edge_benchmark.py || $PY evaluation/edge_benchmark.py ;;
  demo)      $PY python_network/network_monitor.py --dataset-replay "${2:-5000}" ;;
  pcap)      $PY python_network/network_monitor.py --pcap "$2" ;;
  live)      sudo "$PWD/$PY" python_network/network_monitor.py --iface "${2:-any}" ;;
  dashboard) $PY -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.headless true ;;
  *) echo "commands: setup build data experiments train test fixtures benchmark demo [N] pcap FILE live [IFACE] dashboard"; exit 1 ;;
esac
