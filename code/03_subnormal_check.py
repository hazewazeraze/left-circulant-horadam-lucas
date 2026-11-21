# 03_subnormal_check.py
#
# Controlled check for the runtime subsection of the paper: is the jump in
# numpy.linalg.inv between n = 500 and n = 1000 caused by subnormal numbers?
# We time the same inverse twice, once with ordinary IEEE arithmetic and once
# with the CPU told to flush subnormals to zero (FTZ/DAZ). If subnormals are
# the cause, the second timing drops back to roughly cubic growth.
#
# Run from the project folder. Windows needs nothing extra:
#     python code/03_subnormal_check.py --output data/publication
# Linux and macOS (x86-64) need the small helper compiled first:
#     gcc -O2 -shared -fPIC -o code/libftz.so code/ftz.c
#     python code/03_subnormal_check.py --lib code/libftz.so --output data/publication

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"     # one thread, so the mode we set covers the whole LU

import numpy as np
import pandas as pd

# common imports NumPy, so it comes after the environment block above
from common import alt_lucas_float, left_circ


def denormal_switch(lib):
    """Return (flush_on, flush_off) for this platform."""
    if sys.platform == "win32":
        # the C runtime exposes the same SSE control word through _controlfp_s
        crt = ctypes.cdll.msvcrt
        MCW_DN, DN_FLUSH, DN_SAVE = 0x03000000, 0x01000000, 0x00000000
        def set_mode(value):
            current = ctypes.c_uint()
            crt._controlfp_s(ctypes.byref(current), value, MCW_DN)
        return (lambda: set_mode(DN_FLUSH)), (lambda: set_mode(DN_SAVE))
    handle = ctypes.CDLL(lib)
    return handle.ftz_on, handle.ftz_off


def median_time(f, repeats):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description="Time the dense inverse with and without subnormals.")
    ap.add_argument("--lib", default="./libftz.so")
    ap.add_argument("--output", type=Path, default=Path("."))
    ap.add_argument("--repeats", type=int, default=9)
    args = ap.parse_args()
    flush_on, flush_off = denormal_switch(args.lib)

    rows = []
    for n in (200, 500, 700, 1000, 1400):
        A = left_circ(alt_lucas_float(n))
        flush_off()
        t_ieee = median_time(lambda: np.linalg.inv(A), args.repeats)
        X = np.linalg.inv(A)
        flush_on()
        t_ftz = median_time(lambda: np.linalg.inv(A), args.repeats)
        Y = np.linalg.inv(A)
        flush_off()
        # compare after scaling, the raw entries are far outside the norm's range
        m = np.abs(X).max()
        diff = np.linalg.norm((X - Y) / m) / np.linalg.norm(X / m)
        sub = np.mean((X != 0) & (np.abs(X) < np.finfo(float).tiny))
        rows.append(dict(n=n, inv_ieee_median=t_ieee, inv_ftz_median=t_ftz,
                         slowdown=t_ieee / t_ftz, subnormal_fraction=sub, rel_difference=diff))
        print(f"n = {n:5d}  IEEE {t_ieee:.4f} s   FTZ/DAZ {t_ftz:.4f} s   ratio {t_ieee / t_ftz:.1f}")

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "subnormal_check.csv", index=False)


if __name__ == "__main__":
    main()
