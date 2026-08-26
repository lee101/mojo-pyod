"""Load the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYOD_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyod.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpy_knn_distances": ([I] * 10, I),
    "mpy_hbos_score": ([I] * 8 + [F, F], I),
    "mpy_hbos_score_auto": ([I] * 10 + [F, F], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and (
        not os.path.exists(source) or os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        raise BuildError(
            f"compiled library not found at {LIB}; set MOJOPYOD_LIB to its path"
        )
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo executable not found")
    proc = subprocess.run(
        ["bash", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return address


def check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"Mojo {operation} rejected invalid buffer metadata")
