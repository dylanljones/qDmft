# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from scipy.linalg import expm
from qsim.core import *
from qsim import Circuit, Gate

si, sx, sy, sz = pauli


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


def time_evolution_circuit(c, v, step, dtau, u=4):
    u, t = 4, 1
    b_arg = u * dtau / 4
    xy_arg = v * dtau / 2
    for i in range(step):
        c.xy([[1, 2], [3, 4]], xy_arg)
        c.b([1, 3], b_arg)


def measurement(s, v, step, dtau, alpha, beta, n=500):
    c = Circuit(5, 1)
    c.init(s)
    c.h(0)
    c.add_gate(f"c{alpha.upper()}", 1, con=0, trigger=0)
    # time_evolution_circuit(c, v, step, dtau)
    c.add_gate(f"c{beta.upper()}", 1, con=0, trigger=1)
    c.h(0)

    c.run_shot()
    data = np.zeros(n, "complex")
    for i in range(n):
        data[i] = c.state.measure_qubit(c.qubits[0], basis=sy, shadow=True)
    return np.mean(data)


def measure_gf_greater(s, v, step, dtau, verbose=False):
    g1 = measurement(s, v, step, dtau, "x", "x")
    g2 = measurement(s, v, step, dtau, "y", "x")
    g3 = measurement(s, v, step, dtau, "x", "y")
    g4 = measurement(s, v, step, dtau, "y", "y")
    if verbose:
        print("Greater")
        print("xx", g1)
        print("yx", g2)
        print("xy", g3)
        print("yy", g4)
    return -0.25j * (g1 + 1j*g2 - 1j*g3 + g4)


def measure_gf_lesser(s, v, step, dtau, verbose=False):
    g1 = measurement(s, v, step, dtau, "x", "x")
    g2 = measurement(s, v, step, dtau, "x", "y")
    g3 = measurement(s, v, step, dtau, "y", "x")
    g4 = measurement(s, v, step, dtau, "y", "y")
    if verbose:
        print("Lesser")
        print("xx", g1)
        print("xy", g2)
        print("yx", g3)
        print("yy", g4)
    return 0.25j * (g1 - 1j*g2 + 1j*g3 + g4)


def measure_gf(s0, v, n, dtau, verbose=True):
    data = np.zeros(n+1, "complex")
    for step in range(n+1):
        print(f"{step}/{n}")
        gf_g = measure_gf_greater(s0, v, step, dtau, verbose)
        gf_l = measure_gf_lesser(s0, v, step, dtau, verbose)
        data[step] = gf_g - gf_l
    return data


def main():
    u, t, v = 4, 1, 1
    tau_max, n = 6, 6
    dtau = tau_max / n

    s0 = np.load("state.npy")
    data = measure_gf(s0, v, n, dtau)
    plot_measurement(data, dtau)



if __name__ == "__main__":
    main()
