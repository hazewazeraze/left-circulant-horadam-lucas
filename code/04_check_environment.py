# 04_check_environment.py
#
# Prints the interpreter, the package versions and the BLAS actually loaded by
# NumPy, in the form used for the environment paragraph of the paper. Not part
# of the pipeline, so it carries no number:
#     python code/04_check_environment.py

import platform
import sys
from importlib import metadata

PACKAGES = {
    "NumPy": ("numpy",),
    "SciPy": ("scipy",),
    "pandas": ("pandas",),
    "SymPy": ("sympy",),
    "mpmath": ("mpmath",),
    "Matplotlib": ("matplotlib",),
    "threadpoolctl": ("threadpoolctl",),
    # the OpenBLAS wheel ships under one of these names depending on the build
    "scipy-openblas": ("scipy-openblas64", "scipy-openblas32", "scipy-openblas"),
}


def version_of(candidates):
    for name in candidates:
        try:
            return f"{metadata.version(name)}  ({name})"
        except metadata.PackageNotFoundError:
            continue
    return "Not found"


def main():
    print("=== SYSTEM INFORMATION ===")
    print("Python       :", sys.version.split()[0])
    print("OS           :", platform.platform())
    print("Architecture :", platform.architecture()[0])
    print("Processor    :", platform.processor())
    print("Executable   :", sys.executable)

    print("\n=== PACKAGE VERSIONS ===")
    for label, candidates in PACKAGES.items():
        print(f"{label:15}: {version_of(candidates)}")

    print("\n=== BLAS RUNTIME ===")
    try:
        import numpy as np
        np.show_runtime()
    except Exception as exc:
        print("Runtime information unavailable:", exc)


if __name__ == "__main__":
    main()
