# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scitools import Plot
from qsim import pauli, ZERO, kron, Circuit, VqeSolver
from qsim.dmft import gf_greater, gf_lesser
from qsim.dmft import fit_gf_measurement, print_popt, get_gf_fit_data, get_gf_spectral_data
from dmft import TwoSiteSiam, impurity_gf_ref

STATE_FILE = "data/twosite_hf_gs.npy"
DATA_RE_FILE = "data/twosite_hf_re.npz"
DATA_IM_FILE = "data/twosite_hf_im.npz"
DATA_RE_SIM_FILE = "data/twosite_hf_re_sim.npz"
DATA_IM_SIM_FILE = "data/twosite_hf_im_sim.npz"
si, sx, sy, sz = pauli


class MeasurementData(dict):

    def __init__(self, gs=None, times=None, data=None):
        super().__init__(gs=gs, times=times, data=data)

    @property
    def gs(self):
        return self["gs"]

    @property
    def times(self):
        return self["times"]

    @property
    def data(self):
        return self["data"]


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


def _measure(gs, xy_arg, b_arg, step, alpha, beta, imag=True, shots=None):
    c = Circuit(5, 1)
    c.h(0)
    c.add_gate(f"c{alpha.upper()}", qubits=1, con=0, trigger=0)
    for i in range(step):
        c.xy([[1, 2], [3, 4]], xy_arg)
        c.b([1, 3], b_arg)

    c.add_gate(f"c{beta.upper()}", qubits=1, con=0, trigger=1)
    c.h(0)
    c.run_circuit(state=gs)
    if shots is None:
        return c.expectation(sz, 0) if imag else c.expectation(sy, 0)
    else:
        x = np.zeros(shots)
        for i in range(shots):
            x[i] = c.measure_z(0, True)[0] if imag else c.measure_y(0, True)[0]
        return np.mean(x)


def measure_data(siam, gs, nt, tmax, imag=True, shots=None):
    n = nt + 1
    dt = tmax / nt
    b_arg = dt * siam.u / 4
    xy_arg = dt * siam.v / 2
    times = np.arange(n) * dt
    data = np.zeros((n, 4), "complex")
    header = "Measuring " + ("real" if imag is False else "imaginary")
    print(header, end="", flush=True)
    for step in range(n):
        print(f"\r{header}: {100 * (step + 1) / n:.1f}% ({step + 1}/{n})", end="", flush=True)
        xx = _measure(gs, xy_arg, b_arg, step, "x", "x", imag, shots)
        xy = _measure(gs, xy_arg, b_arg, step, "x", "y", imag, shots)
        yx = _measure(gs, xy_arg, b_arg, step, "y", "x", imag, shots)
        yy = _measure(gs, xy_arg, b_arg, step, "y", "y", imag, shots)
        data[step] = [xx, xy, yx, yy]
    print()
    return times, data


def get_measurement_data(siam, gs, nt, tmax, file, imag=True, shots=None, new=False):
    if new or not os.path.isfile(file):
        times, data = measure_data(siam, gs, nt, tmax, imag, shots)
        print("Saving data...")
        np.savez(file, times=times, data=data)
        return times, data
    else:
        print("Loading " + ("real" if imag is False else "imaginary") + " data...")
        file_data = np.load(file)
        return file_data["times"], file_data["data"]


# ========================================================================
#                               GREENS FUNCTION
# ========================================================================


def greens_function(data):
    n = data.shape[0]
    gf = np.zeros(n, dtype="complex")
    for i in range(n):
        xx, xy, yx, yy = data[i]
        gf_g = gf_greater(xx, yx, xy, yy)
        gf_l = gf_lesser(xx, xy, yx, yy)
        gf[i] = gf_g - gf_l
    return gf


def measure_gf_imag(siam, nt, tmax, shots=None, new_state=True, new_data=True):
    state_file = STATE_FILE
    data_file = DATA_IM_FILE
    gs = get_ground_state(siam, new=new_state, file=state_file)
    times, data_im = get_measurement_data(siam, gs, nt, tmax, imag=True, shots=shots, new=new_data, file=data_file)
    return times, -greens_function(data_im).imag


def plot_result(times, data, t_fit, fit, z, gf, gf_ref=None, title=None):
    plot = Plot.subplots(2, 1, hr=(1, 1))
    # plot.set_figsize(width=800)
    plot.add_gridsubplot(0)
    plot.set_limits((0, np.max(times)), (-1.05, 1.05))
    plot.set_labels(r"$\tau t^*$", r"$iG_{imp}^{R}(\tau)$")
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
    u, t = 0, 1
    tmax, nt = 6, 48
    siam = TwoSiteSiam(u=u, eps_imp=0, eps_bath=0, v=t, mu=u/2)
    p0 = [0.5, 0.5, siam.v, siam.u]
    shots = 500

    times, gf_im = measure_gf_imag(siam, nt, tmax, shots, new_data=True, new_state=True)
    popt, errs = fit_gf_measurement(times, gf_im.real, p0=p0)
    t_fit, fit = get_gf_fit_data(popt, tmax, n=100)
    z, gf = get_gf_spectral_data(popt, zmax=4, n=1000)
    print_popt(popt, errs)

    gf_ref = impurity_gf_ref(z, siam.u, siam.v)
    title = f"samples={shots}, N$_t$={nt}"
    plot = plot_result(times, gf_im, t_fit, fit, z, gf, gf_ref, title)
    plt.show()


if __name__ == "__main__":
    main()
