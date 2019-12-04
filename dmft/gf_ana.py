
import numpy as np
from scipy import integrate

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
