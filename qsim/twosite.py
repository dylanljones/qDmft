# -*- coding: utf-8 -*-
"""
Created on 29 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scipy import optimize
from scitools import Terminal, Plot
from qsim import kron, si, sy, sz, Circuit, ZERO


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


def print_popt(popt, dec=2):
    strings = list(["Green's function fit:"])
    strings.append(f"  alpha_1 = {popt[0]:.{dec}}")
    strings.append(f"  alpha_2 = {popt[1]:.{dec}}")
    strings.append(f"  omega_1 = {popt[2]:.{dec}}")
    strings.append(f"  omega_2 = {popt[3]:.{dec}}")
    line = "-" * (max([len(x) for x in strings]) + 1)
    print(line)
    print("\n".join(strings))
    print(line)


def get_gf_fit_data(popt, tmax, n=100):
    t_fit = np.linspace(0, tmax, n)
    return t_fit, fitted_gf(t_fit, popt)


def get_gf_spectral_data(popt, zmax, eta=0.01, n=1000):
    z = np.linspace(-zmax, zmax, n) + 1j * eta
    return z, fitted_gf_spectral(z, popt)


# =========================================================================


def measurement(s, u, v, step, dt, alpha, beta, imag=True, shots=None):
    b_arg = u * dt / 4
    xy_arg = v * dt / 2

    c = Circuit(5, 1)
    c.state.set(s)
    # Configure circuit
    c.h(0)
    c.add_gate(f"c{alpha.upper()}", 1, con=0, trigger=0)
    for i in range(step):
        c.xy([[1, 2], [3, 4]], xy_arg)
        c.b([1, 3], b_arg)
    c.add_gate(f"c{beta.upper()}", 1, con=0, trigger=1)
    c.h(0)
    c.run_shot()
    # Measurement
    if shots is None:
        x = c.expectation(sy, 0) if imag else c.expectation(sz, 0)
    else:
        data = np.zeros(shots, "complex")
        for i in range(shots):
            if imag:
                data[i] = c.state.measure_y(c.qubits[0], shadow=True)[0]
            else:
                data[i] = c.state.measure_z(c.qubits[0], shadow=True)[0]
        x = np.mean(data)
    return x


def measure_gf_greater(s, u, v, step, dt, imag=True, shots=None):
    g1 = measurement(s, u, v, step, dt, "x", "x", imag, shots)
    g2 = measurement(s, u, v, step, dt, "y", "x", imag, shots)
    g3 = measurement(s, u, v, step, dt, "x", "y", imag, shots)
    g4 = measurement(s, u, v, step, dt, "y", "y", imag, shots)
    return gf_greater(g1, g2, g3, g4)


def measure_gf_lesser(s, u, v, step, dt, imag=True, shots=None):
    g1 = measurement(s, u, v, step, dt, "x", "x", imag, shots)
    g2 = measurement(s, u, v, step, dt, "x", "y", imag, shots)
    g3 = measurement(s, u, v, step, dt, "y", "x", imag, shots)
    g4 = measurement(s, u, v, step, dt, "y", "y", imag, shots)
    return gf_lesser(g1, g2, g3, g4)


def measure_gf(s0, u, v, nt, dt, imag=True, shots=None):
    data = np.zeros(nt+1, "complex")
    terminal = Terminal()
    terminal.write("Measuring Green's function")
    for step in range(nt+1):
        terminal.updateln(f"Measuring Green's function: {step}/{nt}")
        gf_g = measure_gf_greater(s0, u, v, step, dt, imag, shots)
        gf_l = measure_gf_lesser(s0, u, v, step, dt, imag, shots)
        data[step] = gf_g - gf_l
    terminal.writeln()
    times = np.arange(len(data)) * dt
    return times, data


def measure_gf_spectral(siam, gs, tmax, nt, z, p0=None, imag=True, shots=None, plot_on_error=True):
    dt = tmax / nt
    gs = kron(ZERO, gs)
    times, data = measure_gf(gs, siam.u, siam.v, nt, dt, imag, shots)
    popt, errs = fit_gf_measurement(times, data.real, p0=p0)
    if np.any(errs >= 1e-5):
        if plot_on_error:
            plot = Plot()
            plot.plot(times, data, color="k", label="Data")
            plot.plot(*get_gf_fit_data(popt, tmax, n=100), color="r", ls="--", label="Fit")
            plot.legend()
            plot.show()
        raise ValueError("Couldn't fit measurement data! Errors:", errs)
    return gf_spectral(z, *popt)
