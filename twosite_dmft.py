# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
from dmft import self_energy, bethe_gf_omega, diagonalize
from dmft.twosite import impurity_gf_ref, impurity_gf_free_ref, TwoSiteSiam
from dmft.twosite import quasiparticle_weight, m2_weight
from dmft.twosite import basis_states, annihilation_operator, HamiltonOperator
from scipy.sparse import csr_matrix
from scitools import Plot
from scitools import plotting

plotting.TEXTWIDTH = 300


def new_hybrid(v, m2, z, mixing=1.0):
    v_new = np.sqrt(z * m2)
    new, old = mixing, 1.0 - mixing
    return (v_new * new) + (v * old)


def solve(z, u, eps_imp, eps_bath, v, mu):
    gf_imp0 = impurity_gf_free_ref(z + mu, eps_imp, eps_bath, v)
    gf_imp = impurity_gf_ref(z, u, v)
    sigma = self_energy(gf_imp0, gf_imp)
    return gf_imp0, gf_imp, sigma


def self_consistency_loop(z, u, eps, t, mu, mixing=1.0, nmax=1000):
    m2 = m2_weight(t)
    eps_imp, eps_bath, v = eps, mu, t
    i = 0
    for i in range(nmax):
        gf_imp0, gf_imp, sigma = solve(z, u, eps_imp, eps_bath, v, mu)
        qp_weight = quasiparticle_weight(z.real, sigma.real)
        new_v = new_hybrid(v, m2, qp_weight, mixing=mixing)
        delta = abs(new_v - v)
        v = new_v
        if delta < 1e-5:
            break
    print(f"Iteration: {i}")
    return solve(z, u, eps_imp, eps_bath, v, mu)


class TwositeDmft:

    def __init__(self, u, eps, t, mu):
        self.u = u
        self.eps = eps
        self.t = t
        self.mu = mu
        self.siam = TwoSiteSiam(u, eps, mu, t, mu)
        self.m2 = m2_weight(t)

        self.gf_imp0 = None
        self.gf_imp = None
        self.sigma = None
        self.gf_latt = None

    def solve_self_consist(self, z, mixing=0.5, thresh=1e-5, nmax=1000):
        v_new = self.siam.v
        i = 0
        for i in range(nmax):
            v = v_new
            self.siam.update_hybridization(v)
            sigma = self.siam.self_energy(z)
            qp_weight = quasiparticle_weight(z.real, sigma)
            v_new = new_hybrid(v, self.m2, qp_weight, mixing=mixing)
            delta = abs(v_new - v)
            if delta < thresh:
                break
        self.siam.update_hybridization(v_new)
        print(f"Iteration: {i}")
        self.gf_imp0 = self.siam.impurity_gf_free(z)
        self.gf_imp = self.siam.impurity_gf(z)
        self.sigma = self_energy(self.gf_imp0, self.gf_imp)
        self.gf_latt = bethe_gf_omega(z + self.mu - self.sigma, self.t)
        return self.gf_imp0, self.gf_imp, self.sigma


def plot_quasiparticle_weight(z, eps, t):
    u_values = np.arange(0, 8, 0.1)
    z_values = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        dmft = TwositeDmft(u, eps, t, u/2)
        gf_imp0, gf_imp, sigma = dmft.solve_self_consist(z)
        z_values[i] = quasiparticle_weight(z.real, sigma)

    fig, ax = plt.subplots()
    ax.plot(u_values, z_values)
    plt.show()


def lattice_greens_function(z, u, eps, t, mu):
    dmft = TwositeDmft(u, eps, t, mu)
    dmft.solve_self_consist(z)
    return dmft.gf_latt


def plot_gf_latt(z, u, eps, t, mu, show=True):
    gf_latt = lattice_greens_function(z, u, eps, t, mu)
    fig, ax = plt.subplots()
    ax.plot(z.real, -gf_latt.imag)
    if show:
        plt.show()


def plot_mott_transition(z, n=50, offset=0.2):
    u_values = np.linspace(0, 8, n)[::-1]
    lines = np.zeros((n, z.size))
    eps, t = 0, 1
    for i, u in enumerate(u_values):
        print(i, u)
        dmft = TwositeDmft(u, eps, t, u/2)
        dmft.solve_self_consist(z)
        lines[i] = -dmft.gf_latt.imag + i * offset

    PATH = r"D:\Dropbox\Uni\Master\Master Thesis\DmftPresentation\img"
    uc = 6
    ic = np.argmin(np.abs(u_values-uc))
    yuc = lines[ic, -1]

    plot = Plot(xlabel=r"$\omega$", ylabel=r"$A_{latt}$")
    plot.latex_plot(width=0.9)
    for i in range(n):
        col = "k" if i != ic else "r"
        plot.plot(z.real, lines[i], color=col, lw=0.5)

    plot.set_ticks(yticks=[])
    # plot.draw_lines(y=yuc, color="r", ls="--", lw=0.5)
    plot.set_limits(xlim=0)

    plot.add_yax()
    plot.set_limits(ylim=plot.axs[0].get_ylim())
    plot.set_ticks(yticks=[lines[0, -1], yuc, lines[-1, -1]])
    plot.set_ticklabels(yticks=[f"U={u_values[0]}", r"U$_C$=6", f"U={u_values[-1]}"])
    plot.show()
    # plot.tight()
    # plot.save(PATH, "mott.png")
    # plot.show()


def main():
    u = 1
    eps, t = 0, 1
    omegas = np.linspace(-6, 6, 10000)
    eta = 0.01j
    z = omegas + eta

    gf_imp = impurity_gf_ref(z, u, t)
    plot = Plot()
    plot.plot(omegas, -gf_imp.imag)
    plot.show()

    # ---------------------------------------------------
    # plot_gf_latt(z, u, eps, t, mu, show=False)
    # plot_quasiparticle_weight(z, eps, t)

    dmft = TwositeDmft(u, eps, t, u/2)
    dmft.solve_self_consist(z)

    plot = Plot()
    plot.plot(omegas, -dmft.gf_latt.imag)
    plot.show()


    # plot_mott_transition(z)


if __name__ == "__main__":
    main()
