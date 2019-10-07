# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: qDmft
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from itertools import product


# =========================================================================
#                               GENERAL
# =========================================================================


def get_eta(omegas, n):
    dw = abs(omegas[1] - omegas[0])
    return 1j * n * dw


def get_omegas(omax=5, n=1000, deta=0.5):
    omegas = np.linspace(-omax, omax, n)
    return omegas, get_eta(omegas, deta)


def spectral(gf):
    return -1/np.pi * gf.imag


def diagonalize(operator):
    """diagonalizes single site Spin Hamiltonian"""
    eig_values, eig_vecs = la.eigh(operator)
    # eig_values -= np.amin(eig_values)
    return eig_values, eig_vecs


def decompose(operator):
    eigvals, eigstates = la.eigh(operator)
    eigstates_adj = np.conj(eigstates).T
    # eigvals -= np.amin(eigvals)
    return eigstates_adj, eigvals, eigstates


def iter_indices(n):
    return product(range(n), range(n))


# =========================================================================
#                               FERMIONS
# =========================================================================


def fermi_dist(energy, beta, mu=1):
    """ Calculates the fermi-distributions for fermions

    Parameters
    ----------
    energy: float nd.ndarray or float
        The energy value
    beta: float
        Coldnes (inverse temperature)
    mu: float, default=0
        Chemical potential. At T=0 this is the Fermi-energy E_F

    Returns
    -------
    fermi: float np.ndarray
    """
    exponent = np.asarray((energy - mu) * beta).clip(-1000, 1000)
    return 1. / (np.exp(exponent) + 1)


def partition_func(beta, energies):
    return np.exp(-beta*energies).sum()


def expectation(eigvals, eigstates, operator, beta):
    ew = np.exp(-beta * eigvals)
    aux = np.einsum('i,ji,ji', ew, eigstates, operator.dot(eigstates))
    return aux / ew.sum()
