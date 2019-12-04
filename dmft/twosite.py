# -*- coding: utf-8 -*-
"""
Created on 4 Dec 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from scipy import optimize

# Reference functions taken from M. Potthof:
# 'Two-site dynamical mean-field theory'


def impurity_gf_free_ref(z, eps0, eps1, v):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + v*v)
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


# ===================================================================================



def self_energy(gf_imp0, gf_imp):
    return 1/gf_imp0 - 1/gf_imp


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
    """ Calculate the filling using the Green's function of the corresponding model"""
    idx = np.argmin(np.abs(omegas)) + 1
    x = omegas[:idx]
    y = -gf[:idx].imag
    x[-1] = 0
    y[-1] = (y[-1] + y[-2]) / 2
    return integrate.simps(y, x)


def new_hybridization(z, m2, v, mixing=1.0):
    v_new = np.sqrt(z * m2)
    new, old = mixing, 1.0 - mixing
    return (v_new * new) + (v * old)


def bethe_dos(z, t):
    """Density of states of the Bethe lattice"""
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


def bethe_gf_omega(z, t=1.0):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    Taken from gf_tools by Weh Andreas
    https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    t : float
        Hopping parameter of the bethe lattice. This defines the bandwidth 'D=4t'
    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function
    """
    half_bandwidth = 2 * t
    z_rel = z / half_bandwidth
    return 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel)))


class TwoSiteSiam:

    def __init__(self, u, eps_imp, eps_bath, v, mu, beta=0.01):
        self.mu = mu
        self.beta = beta
        self.u = float(u)
        self.eps_imp = float(eps_imp)
        self.eps_bath = float(eps_bath)
        self.v = float(v)

    @classmethod
    def half_filling(cls, u, v, eps_imp=0, beta=0.01):
        eps_bath = u / 2
        mu = u / 2
        return cls(u, eps_imp, eps_bath, v, mu, beta)

    def update_bath_energy(self, eps_bath):
        self.eps_bath = float(eps_bath)

    def update_hybridization(self, v):
        self.v = float(v)

    def hybridization(self, z):
        return self.v**2 / (z + self.mu - self.eps_bath)

    def impurity_gf_free(self, z):
        return 1/(z + self.mu - self.eps_imp - self.hybridization(z))

    def new_hybridization():
        pass
