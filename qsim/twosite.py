# -*- coding: utf-8 -*-
"""
Created on 29 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scitools import Plot


def plot_measurement(data, dtau):
    n = len(data)
    tau = np.arange(n) * dtau
    plot = Plot(xlim=[0, np.max(tau)], ylim=[-1, 1])
    plot.set_labels(r"$\tau t^*$", r"$G_{imp}^{R}(\tau)$")
    plot.grid()
    plot.set_title(f"N={n-1}")
    plot.plot(tau, data.real, label="real")
    plot.plot(tau, data.imag, label="imag")
    plot.legend()
    plot.show()


def gf_greater(xx, yx, xy, yy):
    return -0.25j * (xx + 1j*yx - 1j*xy + yy)


def gf_lesser(xx, xy, yx, yy):
    return +0.25j * (xx - 1j*xy + 1j*yx + yy)
