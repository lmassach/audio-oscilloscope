#!/usr/bin/env python3
"""Script to plot various trigger statistics."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import moyal
from uncertainties import ufloat


def fit_exp(x, n, tau):
    return n * np.exp(-x / tau)


def p0est_exp(bins, edges):
    i1, i2 = 1, bins.size//2
    tau = (edges[i1] - edges[i2]) / np.log(bins[i2] / bins[i1])
    n = bins[i1] / np.exp(-edges[i1] / tau)
    return n, tau


def fit_gaus(x, n, mu, sigma):
    return n * np.exp(-0.5 * ((x - mu) / sigma)**2)


def p0est_gaus(bins, edges):
    mean = np.average(edges, weights=bins)
    return np.max(bins), mean, np.sqrt(np.average((edges - mean)**2, weights=bins))


def fit_landau(x, n, mpv, scale):
    return n * moyal.pdf(x, loc=mpv, scale=scale)


def p0est_landau(bins, edges):
    return np.sum(bins), edges[np.argmax(bins)], 1


FIT_FUNCTIONS = {
    'exp': {
        'name': 'exponential',
        'func': fit_exp,
        'p0est': p0est_exp,
        'output': '$\\tau = {1:L}$ s'
    }, 'gaus': {
        'name': 'normal',
        'func': fit_gaus,
        'p0est': p0est_gaus,
        'output': '$\\mu = {1:L}$ s\n$\\sigma = {2:L}$ s'
    }, 'landau': {
        'name': 'Landau',
        'func': fit_landau,
        'p0est': p0est_landau,
        'output': 'MPV $= {1:L}$ s\nWidth $= {2:L}$ s'
    }
}


def plot_and_fit(data, x_label, fit_func=None):
    h, e, _ = plt.hist(data, bins=50, label="Data")
    if fit_func in FIT_FUNCTIONS:
        ff = FIT_FUNCTIONS[fit_func]
        e = (e[:-1] + e[1:]) / 2
        popt, pcov = curve_fit(ff['func'], e, h, p0=ff['p0est'](h, e))
        # print(x_label, ff['p0est'](h, e), popt)
        pstd = np.sqrt(pcov.diagonal())
        chi2 = np.sum((h - ff['func'](e, *popt))**2 / np.maximum(h, 1))
        ndf = len(h) - len(popt)
        label = f"Fit ({ff['name']})\n"
        label += ff['output'].format(*(ufloat(a, b) for a, b in zip(popt, pstd)))
        label += f"\n$\\chi^2$/NDF = {chi2:.3g}/{ndf:.3g}"
        plt.plot(e, ff['func'](e, *popt), '-', label=label)
        plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("Events / bin")
    plt.grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="The output file from oscilloscope's -o option.")
    parser.add_argument("-o", "--output", help="Optional output image file.")
    fit_keys = list(FIT_FUNCTIONS.keys()) + ['none']
    parser.add_argument("--fit-dt", choices=fit_keys, default='exp',
                        help="Fit function for the delta-t distribution (default %(default)s).")
    parser.add_argument("--fit-mean", choices=fit_keys, default='gaus',
                        help="Fit function for the mean distribution (default %(default)s).")
    parser.add_argument("--fit-rms", choices=fit_keys, default='landau',
                        help="Fit function for the RMS distribution (default %(default)s).")
    parser.add_argument("--fit-min", choices=fit_keys, default='none',
                        help="Fit function for the min peak distribution (default %(default)s).")
    parser.add_argument("--fit-max", choices=fit_keys, default='none',
                        help="Fit function for the max peak distribution (default %(default)s).")
    parser.add_argument("--fit-pp", choices=fit_keys, default='landau',
                        help="Fit function for the peak-peak amplitude distribution (default %(default)s).")
    args = parser.parse_args()

    _, trg_time, smp_mean, smp_rms, smp_max, smp_min = np.loadtxt(args.input_file, unpack=True)
    delta_t = np.diff(trg_time)
    delta_t = delta_t[delta_t > 0]

    plt.subplot(231)
    plot_and_fit(delta_t, "$\\Delta t$ between events [s]", args.fit_dt)
    plt.subplot(232)
    plot_and_fit(smp_mean, "Sample mean [au]", args.fit_mean)
    plt.subplot(233)
    plot_and_fit(smp_rms, "Sample RMS [au]", args.fit_rms)
    plt.subplot(234)
    plot_and_fit(smp_min, "Sample minimum peak [au]", args.fit_min)
    plt.subplot(235)
    plot_and_fit(smp_max, "Sample maximum peak [au]", args.fit_max)
    plt.subplot(236)
    plot_and_fit(smp_max - smp_min, "Sample peak-peak amplitude [au]", args.fit_pp)

    if args.output:
        plt.savefig(args.output)
    plt.show()
