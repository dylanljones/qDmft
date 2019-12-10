# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from scitools import prange, Plot
from qsim import pauli, ZERO, kron, Circuit, VqeSolver
from qsim.dmft import fit_gf_measurement, print_popt, get_gf_fit_data, get_gf_spectral_data
from dmft import TwoSiteSiam, impurity_gf_ref

STATE_FILE = "data/siam_gs.npy"
DATA_RE_FILE = "data/measurement_re.npz"
DATA_IM_FILE = "data/measurement_im.npz"

si, sx, sy, sz = pauli

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


def prepare_groundstate(u=4, v=1, eps=None, mu=None):
    print("Preparing ground-state")
    if eps is None:
        eps = u/2
    if mu is None:
        mu = u/2
    ham = hamiltonian(u, eps, mu, v)
    vqe = VqeSolver(ham)
    config_vqe_circuit(vqe.circuit)
    sol = vqe.minimize()
    print(sol)
    return vqe.circuit.state.amp, sol


def get_ground_state(siam, file=STATE_FILE, new=False):
    if new or not os.path.isfile(file):
        gs, sol = prepare_groundstate(siam.u, siam.v, siam.eps_bath, siam.mu)
        if sol.error <= 1e-10:
            print("Saving ground state...")
            np.save(file, gs)
    else:
        print("Loading ground state...")
        gs = np.load(STATE_FILE)
    return kron(ZERO, gs)

# =========================================================================
#                               MEASUREMENT
# =========================================================================


def _measure(gs, xy_arg, b_arg, step, alpha, beta, imag=True):
    c = Circuit(5, 1)
    c.h(0)
    c.add_gate(f"c{alpha.upper()}", qubits=1, con=0, trigger=0)
    for i in range(step):
        c.xy([[1, 2], [3, 4]], xy_arg)
        c.b([1, 3], b_arg)
    c.add_gate(f"c{beta.upper()}", qubits=1, con=0, trigger=1)
    c.h(0)
    c.run_shot(state=gs)
    return c.expectation(sz, 0) if imag else c.expectation(sy, 0)


def measure_data(siam, gs, nt, tmax, imag=True):
    n = nt + 1
    dt = tmax / nt
    b_arg = dt * siam.u / 4
    xy_arg = dt * siam.v / 2
    times = np.arange(n) * dt
    data = np.zeros((n, 4), "complex")
    header = "Measuring " + ("real" if imag is False else "imaginary")
    for step in prange(n, header=header):
        xx = _measure(gs, xy_arg, b_arg, step, "x", "x", imag)
        xy = _measure(gs, xy_arg, b_arg, step, "x", "y", imag)
        yx = _measure(gs, xy_arg, b_arg, step, "y", "x", imag)
        yy = _measure(gs, xy_arg, b_arg, step, "y", "y", imag)
        data[step] = [xx, xy, yx, yy]
    return times, data


def get_measurement_data(siam, gs, nt, tmax, file, imag=True, new=False):
    if new or not os.path.isfile(file):
        times, data = measure_data(siam, gs, nt, tmax, imag)
        print("Saving data...")
        np.savez(file, times=times, data=data)
        return times, data
    else:
        print("Loading " + ("real" if imag is False else "imaginary") + " data...")
        file_data = np.load(file)
        return file_data["times"], file_data["data"]


# =========================================================================
#                               GREENS FUNCTION
# =========================================================================


def gf_greater(xx, yx, xy, yy):
    return -0.25j * (xx + 1j*yx - 1j*xy + yy)


def gf_lesser(xx, xy, yx, yy):
    return +0.25j * (xx - 1j*xy + 1j*yx + yy)


def greens_function(data):
    n = data.shape[0]
    gf = np.zeros(n, dtype="complex")
    for i in range(n):
        xx, xy, yx, yy = data[i]
        gf_g = gf_greater(xx, yx, xy, yy)
        gf_l = gf_lesser(xx, xy, yx, yy)
        gf[i] = gf_g - gf_l
    return gf


def measure_gf_imag(siam, nt, tmax, new_state=True, new_data=True):
    gs = get_ground_state(siam, new=new_state, file=STATE_FILE)
    times, data_im = get_measurement_data(siam, gs, nt, tmax, imag=True, new=new_data, file=DATA_IM_FILE)
    return times, -greens_function(data_im).imag


def plot_result(times, data, t_fit, fit, z, gf, gf_ref=None):
    plot = Plot.subplots(2, 1, hr=(1, 1))
    # plot.set_figsize(width=800)
    plot.add_gridsubplot(0)
    plot.set_limits((0, np.max(times)), (-1.05, 1.05))
    plot.set_labels(r"$\tau t^*$", r"Im $G_{imp}^{R}(\tau)$")
    # plot.plot(times, data.real, label="real", color="k")
    plot.plot(times, data, marker="o", ms=2, label="Data", lw=1, color="k")
    plot.plot(t_fit, fit, color="r", label="Fit", ls="--", lw=1.0)
    plot.legend()
    plot.grid()

    plot.add_gridsubplot(1)
    plot.set_labels(r"$\omega / t^*$", r"-Im $G_{imp}^{R}(\omega + i\eta)$")
    plot.set_limits((np.min(z.real), np.max(z.real)))
    plot.plot(z.real, -gf.imag, color="r", label="Fit")
    if gf_ref is not None:
        plot.plot(z.real, -gf_ref.imag, label="Exact", color="k", ls="--")
    plot.legend()
    plot.grid()
    return plot


def main():
    new_state = False
    new_data = False
    u, t = 4, 1
    tmax, nt = 12, 96
    siam = TwoSiteSiam(u=u, eps_imp=0, eps_bath=0, v=t, mu=u/2)

    times, gf_im = measure_gf_imag(siam, nt, tmax, new_state, new_data)
    popt, errs = fit_gf_measurement(times, gf_im.real, p0=[0.5, 0.5, 1, 4])
    t_fit, fit = get_gf_fit_data(popt, tmax, n=100)
    z, gf = get_gf_spectral_data(popt, zmax=6, n=1000)

    print_popt(popt, errs)
    gf_ref = impurity_gf_ref(z, siam.u, siam.v)
    plot = plot_result(times, gf_im, t_fit, fit, z, gf, gf_ref)
    plot.show()


if __name__ == "__main__":
    main()
