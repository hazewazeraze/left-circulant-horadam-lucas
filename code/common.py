# common.py
#
# Sequences, matrix constructions and the Fourier eigenvalues shared by the
# three scripts of this repository. Everything here is used by more than one
# script, so it lives in one place and has one definition.
#
# Note on import order: 01 and 03 set the BLAS thread variables before any
# NumPy import, and this module imports NumPy, so those scripts import it
# after the environment block.

import math

import numpy as np

ALPHA = (1 + math.sqrt(5)) / 2
EPS = np.finfo(float).eps


def lucas(m):
    """L_m as an exact integer."""
    a, b = 2, 1
    for _ in range(m):
        a, b = b, a + b
    return a


def alt_lucas(n):
    """tilde L_j = (-1)^(j+1) L_j for j < n, as exact integers."""
    return [(-1) ** (j + 1) * lucas(j) for j in range(n)]


def alt_lucas_float(n):
    """The same sequence in float64, built by the recurrence."""
    L = np.empty(n)
    L[0], L[1] = 2.0, 1.0
    for j in range(2, n):
        L[j] = L[j - 1] + L[j - 2]
    return np.where(np.arange(n) % 2 == 0, -L, L)


def left_circ(g):
    # row i of LC_n(g) is g[i], g[i+1], ... (cyclically), i.e. a sliding
    # window over [g, g[:-1]]; one strided copy builds the whole matrix
    g = np.asarray(g)
    n = len(g)
    return np.lib.stride_tricks.sliding_window_view(np.concatenate((g, g[:-1])), n).copy()


def circ(g):
    g = np.asarray(g)
    n = len(g)
    i, j = np.indices((n, n))
    return g[(j - i) % n]


def reversal(n):
    D = np.zeros((n, n))
    D[np.arange(n), (-np.arange(n)) % n] = 1.0
    return D


def fourier_eigs(g):
    # lambda_k = sum_j g_j w^{jk} with w = exp(+2 pi i/n): numpy's fft uses the
    # opposite sign, so we take n * ifft
    g = np.asarray(g)
    return len(g) * np.fft.ifft(g)
