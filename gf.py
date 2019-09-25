# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from itertools import product
from dmft import get_omegas
from dmft.greens import greens_function_free
from dmft.potthof import gf_imp0


def decompose(operator):
    eigvals, eigstates = la.eigh(operator)
    eigstates_adj = np.conj(eigstates).T
    return eigstates_adj, eigvals, eigstates


def greens_function_free2(ham, z):
    """ Calculate the greens function of the given Hamiltonian

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
    z = np.asarray(z)
    # Calculate eigenvalues and -vectors of hamiltonian
    eigstates_adj, eigvals, eigstates = decompose(ham)
    tmat = np.dot(eigstates_adj, eigstates)

    arg = np.subtract.outer(z, eigvals)
    # Calculate greens-function
    n = len(eigvals)
    print(n)
    print(eigvals)
    gf = np.zeros((n, len(z)), dtype="complex")
    for i in range(n):
        for j in range(n):
            gf[i] += np.dot(eigstates_adj[i], eigstates.T[j]) / (z - eigvals[j])

    # subscript_str = "ij,...j,ji->...i"
    # greens = np.einsum(subscript_str, eigstates_adj, 1 / arg, eigstates)
    return gf



def main():
    eps1, eps2, t = 0, 1, 1
    omegas, eta = get_omegas(5, n=1000, deta=0.5)
    ham = np.array([[eps1, t], [t, eps2]])

    fig, ax = plt.subplots()
    ax.plot(omegas, -gf_imp0(omegas + eta, eps1, eps2, t).imag, color="r")
    ax.plot(omegas, -greens_function_free(ham, omegas + eta)[0].imag, color="k", ls="--")

    # ax.plot(omegas, -greens_function_free2(ham, omegas + eta)[0].imag)
    # ax.plot(omegas, -gf_free(ham, omegas + eta)[0].imag)
    plt.show()


if __name__ == "__main__":
    main()
