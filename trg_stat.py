#!/usr/bin/env python3
"""Script to plot various trigger statistics."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat


def fit_exp(x, n, tau):
    return n * np.exp(-x / tau)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="The output file from oscilloscope's -o option.")
    args = parser.parse_args()

    _, trg_time, smp_mean, smp_rms, smp_max, smp_min = np.loadtxt(args.input_file, unpack=True)
    delta_t = np.diff(trg_time)
    delta_t = delta_t[delta_t > 0]

    plt.subplot(231)
    h, e, _ = plt.hist(delta_t, bins=50, label="Data")
    e = (e[:-1] + e[1:]) / 2
    popt, pcov = curve_fit(fit_exp, e, h)
    pstd = np.sqrt(pcov.diagonal())
    chi2 = np.sum((h - fit_exp(e, *popt))**2 / np.maximum(h, 1))
    ndf = len(h) - len(popt)
    plt.plot(e, fit_exp(e, *popt), '-',
             label=f"Fit\n$\\tau = {ufloat(popt[1], pstd[1]):L}$ s\n$\\chi^2$/NDF = {chi2:.3g}/{ndf:.3g}")
    plt.xlabel("$\\Delta t$ between events [s]")
    plt.ylabel("Events / bin")
    plt.legend()
    plt.grid()

    plt.subplot(232)
    plt.hist(smp_mean, bins=50)
    plt.xlabel("Sample mean [au]")
    plt.ylabel("Events / bin")
    plt.grid()

    plt.subplot(233)
    plt.hist(smp_rms, bins=50)
    plt.xlabel("Sample RMS [au]")
    plt.ylabel("Events / bin")
    plt.grid()

    plt.subplot(234)
    plt.hist(smp_min, bins=50)
    plt.xlabel("Sample minimum peak [au]")
    plt.ylabel("Events / bin")
    plt.grid()

    plt.subplot(235)
    plt.hist(smp_max, bins=50)
    plt.xlabel("Sample maximum peak [au]")
    plt.ylabel("Events / bin")
    plt.grid()

    plt.subplot(236)
    plt.hist(smp_max - smp_min, bins=50)
    plt.xlabel("Sample peak-peak amplitude [au]")
    plt.ylabel("Events / bin")
    plt.grid()

    plt.show()
