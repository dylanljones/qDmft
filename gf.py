# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from itertools import product
from dmft import get_omegas
from dmft.greens import greens_function_free
from dmft.potthof import gf_imp0
from dmft.twosite import TwoSiteSiam, annihilate


def time_gfs(siam, z, n=10000):
    ham = siam.hamiltonian_free()
    t0 = time.perf_counter()
    for _ in range(n):
        greens_function_free(ham, z)
    t = time.perf_counter() - t0
    print(f"GF_1: {1000*t/n:.4f} ms")
    t0 = time.perf_counter()
    for _ in range(n):
        greens_function_free2(ham, z)
    t2 = time.perf_counter() - t0
    print(f"GF_2: {1000*t2/n:.4f} ms")


def decompose(operator):
    eigvals, eigstates = la.eigh(operator)
    eigstates_adj = np.conj(eigstates).T
    eigvals -= np.amin(eigvals)
    return eigstates_adj, eigvals, eigstates


def greens_function_free2(ham, z):
    """ Calculates the non-interacting Green's function in the Lehmann representation

    Parameters
    ----------
    ham: array_like
        Hamiltonian matrix
    z: array_like
        Energy e+eta of the greens function.

    Returns
    -------
    greens: np.ndarray
    """
    # Calculate eigenvalues and -vectors of hamiltonian
    # and prepare arguments
    eigstates_adj, eigvals, eigstates = decompose(ham)
    mat = eigstates_adj * eigstates
    arg = np.subtract.outer(z, eigvals)

    # Calculate the diagonal elements of the greens-function
    n = len(eigvals)
    gf = np.zeros((n, len(z)), dtype="complex")
    for i, j in product(range(n), range(n)):
        gf[i] += mat[i, j] / arg[:, j]
    return gf


def greens_function(eigvals, eigstates, operator, z, beta=1.):
    # Create basis and braket matrix <n|c|m> of the given operator
    basis = np.dot(eigstates.T, operator.dot(eigstates))
    qmat = np.square(basis)

    # Calculate the energy gap matrix
    gap = np.add.outer(-eigvals, eigvals)

    # Calculate weights and partition function
    ew = np.exp(-beta*eigvals)
    weights = np.add.outer(ew, ew)
    partition = ew.sum()

    # Construct Green's function
    n = eigvals.size
    gf = np.zeros_like(z)
    for i, j in product(range(n), range(n)):
        gf += qmat[i, j] / (z - gap[i, j]) * weights[i, j]
    return gf / partition


def binstring(val, n):
    return f"{bin(val)[2:]:0>4}"


def greens_function2(ham, states, z, beta):
    # Calculate eigenvalues and -vectors of hamiltonian
    # and prepare arguments
    eigstates_adj, eigvals, eigstates = decompose(ham)
    mat = eigstates_adj * eigstates
    ew = np.exp(-beta * eigvals)
    weights = np.add.outer(ew, ew)         # Exponential weights
    gap = np.add.outer(-eigvals, eigvals)  # Energy gaps
    part = ew.sum()                        # Partition function

    mat = eigstates_adj * eigstates

    # Calculate the interacting Lehmann Green's-function
    n = eigvals.size
    gf = np.zeros_like(z)
    for j in states:
        i = annihilate(j, 0)
        if i is not None:
            x = np.dot(eigstates_adj[i], eigstates[j])
            gf += np.square(x) / (z - (eigvals[j] - eigvals[i]))
            print(binstring(i, 4), binstring(j, 4), f"<{i}|{j}> = {x}")

    return gf


def gf0_hybrid(z, siam):
    delta = siam.hybridization(z)
    return 1/(z + siam.mu + siam.eps_imp - delta)


def main():
    u = 4
    mu = u/2
    beta = 1/10
    eps_imp, eps_bath, v = 0, mu, 1
    omegas, eta = get_omegas(5, n=1000, deta=1)
    z = omegas + eta
    siam = TwoSiteSiam(u, eps_imp, eps_bath, v, mu, beta)
    # ---------------------------------------------------
    # time_gfs(siam, z)

    ham0 = siam.hamiltonian_free()
    ham = siam.hamiltonian().todense()

    gf = greens_function2(ham, siam.states, z + mu, beta)

    fig, ax = plt.subplots()
    ax.plot(omegas, -gf_imp0(z + mu, siam.eps_imp, siam.eps_bath, siam.v).imag, color="r")
    ax.plot(omegas, -greens_function_free(ham0, z + mu)[0].imag, color="k", ls="--")
    ax.plot(omegas, -gf0_hybrid(z, siam).imag)
    # ax.plot(omegas, gf.real)

    plt.show()


if __name__ == "__main__":
    main()
