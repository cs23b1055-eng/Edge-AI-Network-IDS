import json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "integration")); sys.path.insert(0, os.path.join(ROOT, "python_network")); sys.path.insert(0, os.path.dirname(__file__))
PROC = os.path.join(ROOT, "datasets", "processed")

def read_xy(prefix):
    with open(prefix + "_X.bin", "rb") as f:
        n, d = np.fromfile(f, np.int32, 2); X = np.fromfile(f, np.float32).reshape(n, d)
    with open(prefix + "_y.bin", "rb") as f:
        m = np.fromfile(f, np.int32, 1)[0]; y = np.fromfile(f, np.int32)
    assert len(y) == m == n
    return X, y

def write_xy(prefix, X, y):
    with open(prefix + "_X.bin", "wb") as f:
        np.array(X.shape, dtype=np.int32).tofile(f); np.ascontiguousarray(X, np.float32).tofile(f)
    with open(prefix + "_y.bin", "wb") as f:
        np.array([len(y)], dtype=np.int32).tofile(f); np.asarray(y, np.int32).tofile(f)

def train(task, classes, out, hidden="64,32", lr=0.001, batch=256, epochs=15, opt="adam", cw=0.5, seed=42, prefix_dir=PROC, tag=None, patience=4):
    """Run the C++ trainer; returns (log list, seconds)."""
    p = lambda s: os.path.join(prefix_dir, f"{tag or task}_{s}")
    cmd = [os.path.join(ROOT, "build", "ids_train"), "--train-x", p("train_X.bin"), "--train-y", p("train_y.bin"),
           "--val-x", p("val_X.bin"), "--val-y", p("val_y.bin"), "--classes", str(classes), "--hidden", hidden,
           "--lr", str(lr), "--batch", str(batch), "--epochs", str(epochs), "--opt", opt, "--cw-power", str(cw),
           "--seed", str(seed), "--patience", str(patience), "--out", out]
    t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    return [json.loads(l) for l in r.stdout.splitlines()], time.time() - t
