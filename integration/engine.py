"""ctypes bridge to the C++ MLP engine (build/libidsengine.so). Inference only."""
import ctypes, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LIB = os.path.join(ROOT, "build", "libidsengine.so")

class EngineError(RuntimeError):
    pass

class NativeModel:
    def __init__(self, model_path, lib_path=DEFAULT_LIB):
        if not os.path.exists(lib_path):
            raise EngineError(f"engine library not found: {lib_path} (run `make -C cpp_engine`)")
        lib = ctypes.CDLL(lib_path)
        lib.ids_load.restype = ctypes.c_void_p; lib.ids_load.argtypes = [ctypes.c_char_p]
        lib.ids_free.argtypes = [ctypes.c_void_p]
        lib.ids_n_in.restype = ctypes.c_int; lib.ids_n_in.argtypes = [ctypes.c_void_p]
        lib.ids_n_out.restype = ctypes.c_int; lib.ids_n_out.argtypes = [ctypes.c_void_p]
        lib.ids_predict_proba.restype = ctypes.c_int
        lib.ids_predict_proba.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.ids_last_error.restype = ctypes.c_char_p
        lib.ids_real_is_float.restype = ctypes.c_int
        if not lib.ids_real_is_float():
            raise EngineError("engine built with non-float Real; ctypes bridge expects float32")
        self._lib = lib
        self._h = lib.ids_load(os.fsencode(model_path))
        if not self._h:
            raise EngineError(f"cannot load model {model_path}: {lib.ids_last_error().decode()}")
        self.n_in, self.n_out = lib.ids_n_in(self._h), lib.ids_n_out(self._h)
        self.model_path = model_path
        self.size_bytes = os.path.getsize(model_path)

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_in:
            raise ValueError(f"expected (n, {self.n_in}) input, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("input contains NaN/inf")
        out = np.empty((X.shape[0], self.n_out), dtype=np.float32)
        rc = self._lib.ids_predict_proba(self._h, X.ctypes.data, X.shape[0], out.ctypes.data)
        if rc != 0:
            raise EngineError(self._lib.ids_last_error().decode())
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.ids_free(self._h); self._h = None
    def __del__(self):
        try: self.close()
        except Exception: pass
