# -*- coding: utf-8 -*-
"""
Created on 14 Feb 2019
author: Dylan

project: qDmft
version: 1.0
"""
import numpy as np
from itertools import product
from scipy import linalg as la

# =========================================================================


def greens_function_free(ham, z):
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
    eigvals, eigstates = np.linalg.eig(ham)
    eigstates_adj = np.conj(eigstates).T

    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i"
    arg = np.subtract.outer(z, eigvals)
    greens = np.einsum(subscript_str, eigstates_adj, 1 / arg, eigstates)
    return greens.T


def self_energy(gf_imp0, gf_imp):
    return 1/gf_imp0 - 1/gf_imp


def greens_function(eigvals, eigstates, operator, z, beta=1.):
    """Outputs the lehmann representation of the greens function
       omega has to be given, as matsubara or real frequencies"""

    operator = operator.todense()

    ew = np.exp(-beta*eigvals)
    partition = ew.sum()

    basis = np.dot(eigstates.T, operator.dot(eigstates))
    tmat = np.square(basis)
    # tmat *= np.add.outer(ew, ew)
    gap = np.add.outer(-eigvals, eigvals)
    weights = np.add.outer(ew, ew)

    n = eigvals.size
    gf = np.zeros_like(z)
    for i, j in product(range(n), range(n)):
        gf += tmat[i, j] / (z - gap[i, j]) * weights[j, i]
    return gf / partition


# =========================================================================
#                          Transformations
# =========================================================================


def freq_tail_fourier(tail_coef, beta, tau, w_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]

    See also
    --------
    gf_fft

    References
    ----------
    [matsubara] https://en.wikipedia.org/wiki/Matsubara_frequency#Time_Domain
    """
    freq_tail = tail_coef[0] / (1.j * w_n) + tail_coef[1] / (1.j * w_n)**2 + tail_coef[2] / (1.j * w_n)**3
    time_tail = - tail_coef[0] / 2 + tail_coef[1] / 2 * (tau - beta / 2) - tail_coef[2] / 4 * (tau**2 - beta * tau)
    return freq_tail, time_tail


def gf_tau_fft(gf_tau, tau, omega, tail_coef=(1., 0., 0.)):
    """ Perform a fourier transform on the imaginary time Green's function

    Parameters
    ----------
    gf_tau : real float array
        Imaginary time green's function to transform
    tau : real float array
        Imaginary time points
    omega : real float array
        fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    gf_omega: complex ndarray or complex
        Value of the transformed imaginary frequency green's function
    """
    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, omega)

    gf_tau = gf_tau - time_tail
    gf_omega = beta * np.fft.ifft(gf_tau * np.exp(1j * np.pi * tau / beta))[..., :len(omega)] + freq_tail
    return gf_omega


def gf_omega_fft(gf_omega, tau, omega, tail_coef=(1., 0., 0.)):
    """ Perform a fourier transform on the imaginary frequency Green's function

    Parameters
    ----------
    gf_omega : real float array
        Imaginary frequency Green's function to transform
    tau : real float array
        Imaginary time points
    omega : real float array
        fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    gf_tau: complex ndarray or complex
        Value of the transformed imaginary time green's function
    """
    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, omega)

    gf_omega = gf_omega - freq_tail
    gf_tau = np.fft.fft(gf_omega, len(tau)) * np.exp(-1j * np.pi * tau / beta)
    gf_tau = (2 * gf_tau / beta).real + time_tail
    return gf_tau