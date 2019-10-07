# -*- coding: utf-8 -*-
"""
Created on 19 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from scipy import optimize
from scipy.sparse import csr_matrix
from itertools import product
from .greens import self_energy
from .bethe import bethe_dos
from .utils import diagonalize


# Reference functions taken from M. Potthof:
# 'Two-site dynamical mean-field theory'


def impurity_gf_free_ref(z, eps0, eps1, t):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + t*t)
    term1 = (r - e) / (z - e - r)
    term2 = (r + e) / (z - e + r)
    return 1/(2*r) * (term1 + term2)


# Reference functions taken from E. Lange:
# 'Renormalized vs. unrenormalized perturbation-theoretical
# approaches to the Mott transition'

def impurity_gf_ref(z, u, v):
    sqrt16 = np.sqrt(u ** 2 + 16 * v ** 2)
    sqrt64 = np.sqrt(u ** 2 + 64 * v ** 2)
    a1 = 1/4 * (1 - (u ** 2 - 32 * v ** 2) / np.sqrt((u ** 2 + 64 * v ** 2) * (u ** 2 + 16 * v ** 2)))
    a2 = 1/2 - a1
    e1 = 1/4 * (sqrt64 - sqrt16)
    e2 = 1/4 * (sqrt64 + sqrt16)
    return (a1 / (z - e1) + a1 / (z + e1)) + (a2 / (z - e2) + a2 / (z + e2))


# =========================================================================


def m2_weight(t):
    """ Calculates the second moment weight

    Parameters
    ----------
    t: float
        Hopping parameter of the lattice model

    Returns
    -------
    m2: float
    """
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    """ Calculates the quasiparticle weight

    Parameters
    ----------
    omegas: array_like
        Array containig frequency values
    sigma: array_like
        Array containig self energy values
    Returns
    -------
    z: float
    """
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= dw)
    dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
    z = 1/(1 - dsigma)
    if z < 0.01:
        z = 0
    return z


def filling(omegas, gf):
    """ Calculate the filling using the Green's function of the corresponding model

    Parameters
    ----------
    omegas: array_like
        Array containig frequency values
    gf: array_like
        Array containig the Green's function values

    Returns
    -------
    n: float
    """
    idx = np.argmin(np.abs(omegas)) + 1
    x = omegas[:idx]
    y = -gf[:idx].imag
    x[-1] = 0
    y[-1] = (y[-1] + y[-2]) / 2
    return integrate.simps(y, x)


# =========================================================================
# Basis states and operators
# =========================================================================


def basis_states(sites):
    """ Creates basis states for a many-body system in binary representation

    The states are initialized as integer. The binary represents the occupation of the lattice.

    idx     3↓ 3↑ 2↓ 2↑ 1↓ 1↑
    binary  0  1  0  1  1  0

    Parameters
    ----------
    sites: int
        Number of sites in the system
    Returns
    -------
    states: list of int
    """
    n = 2 ** (2 * sites)
    return list(range(n))


def annihilate(state, i):
    """ Act annihilation operator on state

    Parameters
    ----------
    state: int
        Many-body state in binary representation
    i: int
        Index of annihilation operator

    Returns
    -------
    state: int
        Annihilated state
    """
    if not int(state >> i) & 1:
        return None
    return state ^ (1 << i)


def phase(state, i):
    """Phase for fermionic operators"""
    particles = bin(state >> i + 1).count("1")
    return 1 if particles % 2 == 0 else -1


def annihilation_operator(states, idx):
    """ Create annihilation operator in matrix representation for a given set of basis states

    Parameters
    ----------
    states: list_like of int
        Basis states of the system
    idx: int
        Index of annihilation operator

    Returns
    -------

    """
    n = len(states)
    row, col, data = list(), list(), list()
    for j in range(n):
        state = states[j]
        other = annihilate(state, idx)
        if other is not None:
            i = states.index(other)
            val = phase(state, idx)
            row.append(i)
            col.append(j)
            data.append(val)
    return csr_matrix((data, (row, col)), shape=(n, n), dtype="int")


class HamiltonOperator:

    def __init__(self, operators):
        c0u, c0d, c1u, c1d = operators
        self.u_op = c0u.T * c0u * c0d.T * c0d
        self.eps_imp_op = c0u.T * c0u + c0d.T * c0d
        self.eps_bath_op = c1u.T * c1u + c1d.T * c1d
        self.v_op = (c0u.T * c1u + c1u.T * c0u) + (c0d.T * c1d + c1d.T * c0d)

    def build(self, u=5., eps_imp=0., eps_bath=0., v=1.):
        return u * self.u_op + eps_imp * self.eps_imp_op + eps_bath * self.eps_bath_op + v * np.abs(self.v_op)


class TwoSiteSiam:

    def __init__(self, u, eps_imp, eps_bath, v, mu, beta=0.01):
        self.mu = mu
        self.beta = beta
        self.u = float(u)
        self.eps_imp = float(eps_imp)
        self.eps_bath = float(eps_bath)
        self.v = float(v)

        self.states = basis_states(2)
        self.ops = [annihilation_operator(self.states, i) for i in range(4)]
        self.ham_op = HamiltonOperator(self.ops)
        self.eig = None

    def update_bath_energy(self, eps_bath):
        self.eps_bath = float(eps_bath)

    def update_hybridization(self, v):
        self.v = float(v)

    def param_str(self, dec=2):
        parts = [f"u={self.u:.{dec}}", f"eps_imp={self.eps_imp:.{dec}}",
                 f"eps_bath={self.eps_bath:.{dec}}", f"v={self.v:.{dec}}"]
        return ", ".join(parts)

    def bathparam_str(self, dec=2):
        parts = [f"eps_bath={self.eps_bath:.{dec}}", f"v={self.v:.{dec}}"]
        return ", ".join(parts)

    def hybridization(self, z):
        delta = self.v ** 2 / (z + self.mu - self.eps_bath)
        return delta

    def hamiltonian(self):
        return self.ham_op.build(self.u, self.eps_imp, self.eps_bath, self.v)

    def diagonalize(self):
        ham_sparse = self.hamiltonian()
        self.eig = diagonalize(ham_sparse.todense())

    def impurity_gf(self, z, spin=0):
        # self.diagonalize()
        # eigvals, eigstates = self.eig
        # return greens_function(eigvals, eigstates, self.ops[spin].todense(), z + self.mu, self.beta)
        return impurity_gf_ref(z, self.u, self.v)

    def impurity_gf_free(self, z):
        # return 1/(z + self.mu - self.eps_imp - self.hybridization(z))
        return impurity_gf_free_ref(z + self.mu, self.eps_imp, self.eps_bath, self.v)

    def self_energy(self, z):
        gf_imp0 = self.impurity_gf_free(z)
        gf_imp = self.impurity_gf(z)
        return self_energy(gf_imp0, gf_imp)

    def __str__(self):
        return f"Siam({self.param_str()})"
