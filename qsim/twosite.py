# -*- coding: utf-8 -*-
"""
Created on 29 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scipy import optimize


def gf_greater(xx, yx, xy, yy):
    return -0.25j * (xx + 1j*yx - 1j*xy + yy)


def gf_lesser(xx, xy, yx, yy):
    return +0.25j * (xx - 1j*xy + 1j*yx + yy)


def gf_fit(t, alpha_1, alpha_2, omega_1, omega_2):
    return 2 * (alpha_1 * np.cos(omega_1 * t) + alpha_2 * np.cos(omega_2 * t))


def fit_gf_measurement(t, data, p0=None, alpha_max=1, omega_max=100):
    bounds = (0, [alpha_max, alpha_max, omega_max, omega_max])
    popt, pcov = optimize.curve_fit(gf_fit, t, data, p0=p0, bounds=bounds)
    errs = np.sqrt(np.diag(pcov))
    return popt, errs


def fitted_gf(t_fit, popt):
    return gf_fit(t_fit, *popt)


def gf_spectral(z, alpha_1, alpha_2, omega_1, omega_2):
    t1 = alpha_1 * (1 / (z + omega_1) + 1 / (z - omega_1))
    t2 = alpha_2 * (1 / (z + omega_2) + 1 / (z - omega_2))
    return t1 + t2


def fitted_gf_spectral(z, popt):
    return gf_spectral(z, *popt)
