# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from scipy.linalg import expm
from scitools import Terminal
from qsim.core import *
from qsim import Circuit, VqeSolver
from qsim.twosite import *

si, sx, sy, sz = pauli

STATE_FILE = "siam_gs.npy"


def plot_measurement(tau, data, fit=None):
    plot = Plot(xlim=[0, np.max(tau)], ylim=[-1, 1])
    plot.set_labels(r"$\tau t^*$", r"$G_{imp}^{R}(\tau)$")
    plot.grid()
    plot.set_title(f"N={len(data)-1}")
    plot.plot(tau, data.real, label="real", color="k")
    plot.plot(tau, data.imag, label="imag", lw=0.5)
    if fit is not None:
        plot.plot(*fit, color="r", label="Fit")
    plot.legend()
    plot.show()


def plot_all(tau, data, t_fit, fit, z, gf):
    plot = Plot.subplots(2, 1)
    plot.add_gridsubplot(0)
    plot.set_limits((0, np.max(tau)), (-1.05, 1.05))
    plot.set_labels(r"$\tau t^*$", r"$G_{imp}^{R}(\tau)$")
    plot.plot(tau, data.real, label="real", color="k")
    plot.plot(tau, data.imag, label="imag", lw=0.5)
    plot.plot(t_fit, fit, color="r", label="Fit", ls="--")
    plot.legend()
    plot.grid()

    plot.add_gridsubplot(1)
    plot.set_labels(r"$\omega$", r"$A(\omega)$")
    plot.set_limits((np.min(z.real), np.max(z.real)))
    plot.plot(z.real, -gf.imag)
    plot.grid()
    plot.show()


# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def hamiltonian(u=4, eps=2, mu=2, v=1):
    u_op = 1/2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    mu_op = (kron(sz, si, si, si) + kron(si, si, sz, si))
    eps_op = (kron(si, sz, si, si) + kron(si, si, si, sz))
    v_op = kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy)
    ham = u*u_op + mu*mu_op - eps*eps_op + v*v_op
    return 1/2 * ham.real


def config_vqe_circuit(c):
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def prepare_groundstate(file, u=4, v=1, eps=None, mu=None):
    if eps is None:
        eps = u/2
    if mu is None:
        mu = u/2
    ham = hamiltonian(u, eps, mu, v)
    vqe = VqeSolver(ham)
    config_vqe_circuit(vqe.circuit)
    bounds = [(0, 2*np.pi)] * vqe.n_params
    sol = vqe.minimize(bounds=bounds)
    print(sol)
    if sol.error <= 1e-10:
        vqe.save_state(file)


# =========================================================================
#                               MEASUREMENT
# =========================================================================


def time_evolution_circuit(circ, u, v, step, dtau):
    b_arg = u * dtau / 4
    xy_arg = v * dtau / 2
    for i in range(step):
        circ.xy([[1, 2], [3, 4]], xy_arg)
        circ.b([1, 3], b_arg)


def measurement(s, u, v, step, dtau, alpha, beta, n=None):
    c = Circuit(5, 1)
    c.state.set(s)
    c.h(0)
    c.add_gate(f"c{alpha.upper()}", 1, con=0, trigger=0)
    time_evolution_circuit(c, u, v, step, dtau)
    c.add_gate(f"c{beta.upper()}", 1, con=0, trigger=1)
    c.h(0)

    c.run_shot()
    if n is None:
        x = c.expectation(sy, 0)
    else:
        data = np.zeros(n, "complex")
        for i in range(n):
            data[i] = c.state.measure_y(c.qubits[0], shadow=True)[0]
        x = np.mean(data)
    return x


def measure_gf_greater(s, u, v, step, dtau, n=None, verbose=False):
    g1 = measurement(s, u, v, step, dtau, "x", "x", n)
    g2 = measurement(s, u, v, step, dtau, "y", "x", n)
    g3 = measurement(s, u, v, step, dtau, "x", "y", n)
    g4 = measurement(s, u, v, step, dtau, "y", "y", n)
    if verbose:
        print("Greater")
        print("xx", g1)
        print("yx", g2)
        print("xy", g3)
        print("yy", g4)
    return gf_greater(g1, g2, g3, g4)


def measure_gf_lesser(s, u, v, step, dtau, n=None, verbose=False):
    g1 = measurement(s, u, v, step, dtau, "x", "x", n)
    g2 = measurement(s, u, v, step, dtau, "x", "y", n)
    g3 = measurement(s, u, v, step, dtau, "y", "x", n)
    g4 = measurement(s, u, v, step, dtau, "y", "y", n)
    if verbose:
        print("Lesser")
        print("xx", g1)
        print("xy", g2)
        print("yx", g3)
        print("yy", g4)
    return gf_lesser(g1, g2, g3, g4)


def measure_gf(s0, u, v, n, dtau, shots=None, verbose=True):
    data = np.zeros(n+1, "complex")
    terminal = Terminal()
    terminal.write("Measuring Green's function")
    for step in range(n+1):
        terminal.updateln(f"Measuring Green's function: {step}/{n}")
        gf_g = measure_gf_greater(s0, u, v, step, dtau, shots, verbose)
        gf_l = measure_gf_lesser(s0, u, v, step, dtau, shots, verbose)
        data[step] = gf_g - gf_l
    terminal.writeln()
    tau = np.arange(len(data)) * dtau
    return tau, data


# =========================================================================


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


def measure_greens(gs, u, v, tau_max, n):
    dtau = tau_max / n

    gs = kron(ZERO, gs)
    tau, data = measure_gf(gs, u, v, n, dtau, shots=None, verbose=False)

    popt, errs = fit_gf_measurement(tau, data.real, p0=[0.1, 0.4, 1, 2.5])
    print_popt(popt)
    t_fit = np.linspace(0, tau_max, 100)
    fit = fitted_gf(t_fit, popt)
    z = np.linspace(-6, 6, 1000) + 0.01j
    gf = fitted_gf_spectral(z, popt)

    # Plotting
    plot_all(tau, data, t_fit, fit, z, gf)
    # plot_measurement(tau, data, fit)


def main():
    u, t, v = 4, 1, 1
    tau_max, n = 6, 24

    prepare_groundstate(STATE_FILE, u, v)
    gs = np.load(STATE_FILE)
    measure_greens(gs, u, v, tau_max, n)


if __name__ == "__main__":
    main()
