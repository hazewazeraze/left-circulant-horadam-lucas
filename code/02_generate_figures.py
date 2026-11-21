# 02_generate_figures.py
#
# Draws Figures 1-5 of the paper from the CSV files written by
# 01_generate_table_data.py:
#     python code/02_generate_figures.py --data data/publication --output figures
# Add --diagnostics for an extra accuracy plot that is not in the paper.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from common import ALPHA, alt_lucas_float, circ, fourier_eigs, left_circ, reversal

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua", "TeX Gyre Pagella", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

# navy / maroon / teal / gold read well in print; series also differ by
# marker and line style, so the plots survive grayscale
NAVY, MAROON, TEAL, GOLD = "#1F3A5F", "#8C2F39", "#1B6F6F", "#B8860B"
INK, GRID = "#26241F", "#B9B2A2"
FILLS = ("#F4F1EA", "#E6E0D2", "#D6CDB8")


def int_label(x):
    return rf"${int(round(float(x)))}$"


def dec_label(x, digits=2):
    return rf"${float(x):.{digits}f}$"


def tick(x, _):
    return rf"${int(round(x))}$" if np.isclose(x, round(x), atol=1e-10) else rf"${x:g}$"


def finish(ax, ylabel, legend_cols):
    ax.set_xlabel(r"Matrix order $n$")
    ax.set_ylabel(ylabel)
    ax.grid(which="major", linewidth=0.4, alpha=0.6, color=GRID)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=legend_cols,
              frameon=False, handletextpad=0.5, columnspacing=1.2)


def save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)


# Figure 1 -- Delta_4 x C_4 = LC_4

def fig_factorization(out):
    n = 4
    g = alt_lucas_float(n)
    mats = [reversal(n), circ(g), left_circ(g)]
    titles = [r"$\Delta_4$", r"$\mathcal{C}_4\left(\widetilde L\right)$",
              r"$\mathcal{LC}_4\left(\widetilde L\right)$"]

    # the x and = signs get their own narrow grid columns; placing them with
    # fig.text came out off-centre once set_aspect("equal") shrank the axes
    fig = plt.figure(figsize=(8.6, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.5, 3, 0.5, 3], wspace=0.08)
    axes = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4)]
    ops = [fig.add_subplot(gs[0, c]) for c in (1, 3)]

    for ax, M, title, fill in zip(axes, mats, titles, FILLS):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fill,
                                       edgecolor=GRID, linewidth=0.7))
                ax.text(j, i, int_label(M[i, j]), ha="center", va="center", fontsize=9.2, color=INK)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color(INK)
        ax.set_title(title, pad=7, color=INK)

    for ax, sym in zip(ops, (r"$\times$", r"$=$")):
        ax.set_axis_off()
        ax.text(0.5, 0.5, sym, ha="center", va="center", fontsize=17, transform=ax.transAxes, color=INK)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.04)
    save(fig, out / "fig_factorization_lc4.pdf")


# Figure 2 -- spectrum of LC_10

def spread(targets, gap):
    # 1-D label placement: overlapping labels are merged into blocks and each
    # block is centred on the mean of its targets, until nothing overlaps.
    # This keeps every label as close to its point as the spacing allows.
    order = np.argsort(targets)
    blocks = [[targets[i], [i]] for i in order]          # [centre, members]
    merged = True
    while merged:
        merged = False
        for b in range(len(blocks) - 1):
            left, right = blocks[b], blocks[b + 1]
            left_end = left[0] + gap * (len(left[1]) - 1) / 2
            right_start = right[0] - gap * (len(right[1]) - 1) / 2
            if right_start - left_end < gap:
                members = left[1] + right[1]
                blocks[b:b + 2] = [[np.mean(targets[members]), members]]
                merged = True
                break
    pos = np.empty(len(targets))
    for centre, members in blocks:
        offsets = gap * (np.arange(len(members)) - (len(members) - 1) / 2)
        pos[members] = centre + offsets
    return pos


def fig_spectrum(out):
    n = 10
    lam = fourier_eigs(alt_lucas_float(n))
    mods = np.abs(lam[1:n // 2])

    # value, which eigenvalue it is, and whether it belongs to a +/- pair
    points = [(lam[0].real, r"$\lambda_0$", False), (lam[n // 2].real, r"$\lambda_5$", False)]
    for k, m in enumerate(mods, start=1):
        points += [(m, rf"$+|\lambda_{k}|$", True), (-m, rf"$-|\lambda_{k}|$", True)]
    xs = np.array([p[0] for p in points])
    paired = np.array([p[2] for p in points])

    # the markers sit exactly on the real axis; every label sits on one
    # common row, spread just enough not to overlap, with an elbow leader
    # going straight up from the point before bending towards its label
    lo, hi = xs.min() - 18, xs.max() + 18
    slots = spread(xs, 0.088 * (hi - lo))
    y_bend, y_label = 0.30, 0.62

    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axhline(0, linewidth=0.8, color=INK, zorder=1)
    for x, sx in zip(xs, slots):
        ax.plot([x, x, sx], [0.07, y_bend, y_label - 0.04], color=GRID, linewidth=0.6, zorder=1)
    ax.scatter(xs[paired], np.zeros(paired.sum()), marker="o", facecolors="white",
               edgecolors=NAVY, linewidths=1.1, s=34, zorder=3,
               label=r"Paired eigenvalues $\pm\left|\lambda_k\right|$, $k=1,\ldots,4$")
    ax.scatter(xs[~paired], np.zeros((~paired).sum()), marker="s", color=MAROON, s=34, zorder=4,
               label=r"Unpaired eigenvalues $\lambda_0$, $\lambda_5$")
    for (x, name, is_pair), sx in zip(points, slots):
        value = dec_label(x) if is_pair else int_label(x)
        ax.text(sx, y_label, value, ha="center", va="bottom", fontsize=7.4, color=INK)
        ax.text(sx, y_label + 0.27, name, ha="center", va="bottom", fontsize=7.2,
                color=MAROON if not is_pair else NAVY)

    ax.set_xlim(min(lo, slots.min() - 14), max(hi, slots.max() + 14))
    ax.set_ylim(-0.10, y_label + 0.62)
    ax.set_yticks([])
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.10))
    ax.spines["bottom"].set_color(INK)
    ax.set_xlabel("Eigenvalue")
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out / "fig_spectrum_lc10.pdf")


# Figure 3 -- speed-ups relative to the dense numpy routines

def fig_speedup(data, out):
    d = pd.read_csv(data / "figure_speedup.csv")
    n = d["n"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.loglog(n, d["det_speedup"], "o-", color=NAVY, label="Determinant, FFT")
    ax.loglog(n, d["det_closed_speedup"], "o--", color=NAVY, markerfacecolor="white",
              label="Determinant, closed form")
    ax.loglog(n, d["eig_speedup"], "^-.", color=MAROON, label="Spectrum, FFT")
    ax.loglog(n, d["inv_speedup"], "s-", color=TEAL, label="Full inverse, FFT")
    ax.loglog(n, d["inv_scipy_speedup"], "s:", color=GOLD, markerfacecolor="white",
              label="Full inverse, solve_circulant")
    ax.axhline(1, color="0.35", linestyle=":", linewidth=1)
    finish(ax, r"Median speedup $t_{\mathrm{dense}}/t$", 2)
    fig.tight_layout()
    save(fig, out / "fig_speedup.pdf")


# Figure 4 -- the four ways of getting the inverse

def fig_inverse(data, out):
    d = pd.read_csv(data / "figure_inverse_representations.csv")
    n = d["n"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.loglog(n, d["inv_standard_median"], "o-", color=MAROON, label="Dense full inverse")
    ax.loglog(n, d["inv_scipy_circulant_median"], "v:", color=GOLD, label="solve_circulant full inverse")
    ax.loglog(n, d["inv_spectral_median"], "s--", color=TEAL, label="Spectral full inverse")
    ax.loglog(n, d["inv_generator_median"], "^-.", color=NAVY, label="Spectral inverse generator")
    finish(ax, "Median computation time (s)", 2)
    fig.tight_layout()
    save(fig, out / "fig_inverse_representations.pdf")


# Figure 5 -- |kappa_2 - alpha^3| against the rates of the condition-number theorem

def fig_condition(data, out):
    d = pd.read_csv(data / "figure_condition_deviation.csv")
    d = d[~d["singular"]]
    floor = np.finfo(float).eps
    dev = np.maximum(d["absolute_difference_from_alpha3"], floor)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for parity, marker, color in (("even", "o", NAVY), ("odd", "s", MAROON)):
        m = d["parity"] == parity
        ax.loglog(d["n"][m], dev[m], marker=marker, linestyle="none", markersize=3.6,
                  color=color, label=f"{parity.capitalize()} $n$ (computed)")

    ne = np.arange(6, 90, 1.0)
    ne = ne[(5 + 3 * np.sqrt(5)) * ALPHA ** (2 - ne) > floor / 4]   # stop just under epsilon
    ax.loglog(ne, (5 + 3 * np.sqrt(5)) * ALPHA ** (2 - ne), color=NAVY, linewidth=1,
              label=r"$(5+3\sqrt{5})\,\alpha^{2-n}$")
    no = np.geomspace(5, d["n"].max(), 200)
    ax.loglog(no, ALPHA ** 6 * np.pi ** 2 / (2 * no ** 2), "--", color=MAROON, linewidth=1,
              label=r"$\alpha^6\pi^2/(2n^2)$")

    ax.axhline(floor, color=INK, linestyle=":", linewidth=0.9, zorder=2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.35, hi)
    # small n is still empty space, so the epsilon label goes there
    ax.annotate("Float64 epsilon", xy=(d["n"].min(), floor), xytext=(4, 6),
                textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel(r"Matrix order $n$")
    ax.set_ylabel(r"Absolute deviation $\left|\kappa_2-\alpha^3\right|$")
    ax.grid(which="major", linewidth=0.4, alpha=0.6, color=GRID)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
              handletextpad=0.5, columnspacing=1.0)
    fig.tight_layout()
    save(fig, out / "fig_condition_number_deviation.pdf")


# not in the paper -- only for looking at the numerics

def fig_accuracy(data, out):
    d = pd.read_csv(data / "table_accuracy.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for col, label, style in (("logabsdet_rel_error_exact_fft", "Determinant (FFT vs exact)", "o-"),
                              ("inverse_rel_fro_error", "Full inverse", "s--"),
                              ("spectrum_rel_l2_error", "Spectrum", "^-.")):
        ax.loglog(d["n"], np.maximum(d[col], np.finfo(float).eps), style, label=label)
    finish(ax, "Relative error", 3)
    fig.tight_layout()
    save(fig, out / "fig_numerical_accuracy_diagnostic.pdf")


def main():
    ap = argparse.ArgumentParser(description="Draw the figures of the paper.")
    ap.add_argument("--data", type=Path, default=Path("data/publication"))
    ap.add_argument("--output", type=Path, default=Path("figures"))
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args()

    needed = ["figure_speedup.csv", "figure_inverse_representations.csv",
              "figure_condition_deviation.csv", "table_accuracy.csv"]
    missing = [f for f in needed if not (args.data / f).exists()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in {args.data.resolve()} "
                         "-- run 01_generate_table_data.py first")

    args.output.mkdir(parents=True, exist_ok=True)
    fig_factorization(args.output)
    fig_spectrum(args.output)
    fig_speedup(args.data, args.output)
    fig_inverse(args.data, args.output)
    fig_condition(args.data, args.output)
    if args.diagnostics:
        fig_accuracy(args.data, args.output)
    print(f"figures written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
