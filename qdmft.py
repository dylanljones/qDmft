# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
import numpy.linalg as la
from scitools import Terminal
from qsim.core import *
from qsim import Circuit, VqeSolver
from qsim.twosite import *
from dmft import *

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
    plot.set_figsize(width=800)
    plot.add_gridsubplot(0)
    plot.set_limits((0, np.max(tau)), (-1.05, 1.05))
    plot.set_labels(r"$\tau t^*$", r"$G_{imp}^{R}(\tau)$")
    plot.plot(tau, data.real, label="real", color="k")
    plot.plot(tau, data.imag, label="imag", lw=0.5)
    plot.plot(t_fit, fit, color="r", label="Fit", ls="--", lw=1.0)
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
    print("Preparing ground-state")
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
    return vqe.circuit.state.amp


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


def plot_measurement(siam, gs, tmax, nt):
    dt = tmax / nt
    gs = kron(ZERO, gs)
    times, data = measure_gf(gs, siam.u, siam.v, nt, dt, imag=True, shots=None)
    popt, errs = fit_gf_measurement(times, data.real, p0=[0.1, 0.4, 1, 2.5])
    print_popt(popt)
    t_fit, fit = get_gf_fit_data(popt, tmax, n=100)
    z, gf = get_gf_spectral_data(popt, 6, n=1000)

    plot_all(times, data, t_fit, fit, z, gf)



def main():
    u, t = 4, 1
    tmax, nt = 6, 24
    omax = 4
    omegas = np.linspace(-omax, omax, 10000)
    z = omegas + 0.01j
    p0 = [0.1, 0.4, 1, 2.5]

    siam = TwoSiteSiam.half_filling(u=u, v=t)
    m2 = m2_weight(t)

    # gs = prepare_groundstate(STATE_FILE, u, v)
    gs = np.load(STATE_FILE)

    gf = measure_gf_spectral(siam, gs, tmax, nt, z, p0)
    gf_ref = impurity_gf_ref(z, siam.u, siam.v)

    plot = Plot()
    plot.plot(z.real, -gf.imag, label="Measured")
    plot.plot(z.real, -gf_ref.imag, label="Exact")
    plot.legend()
    plot.show()


    return
    gf_0 = impurity_gf_free_ref(z, siam.eps_imp, siam.eps_bath, siam.v)
    sigma = self_energy(gf_0, gf)

    qp = quasiparticle_weight(z.real, sigma)
    v_new = new_hybridization(qp, m2, siam.v)
    print(v_new)

    plot = Plot(xlabel=r"$\omega$")
    plot.grid()
    plot.plot(z.real, -sigma.imag, color="k", label=r"-Im $\Sigma_{imp}(z)$")
    plot.plot(z.real, -gf_0.imag, label=r"-Im $G_{imp}^{0}(z)$")
    plot.plot(z.real, -gf.imag, label=r"-Im $G_{imp}(z)$")

    plot.set_limits(0)
    plot.legend()

    plot.show()






if __name__ == "__main__":
    main()
