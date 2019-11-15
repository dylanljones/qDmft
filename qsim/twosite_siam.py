# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from .fermions import FBasis, HamiltonOperator
from .core import pauli, kron

si, sx, sy, sz = pauli


def twosite_basis():
    return FBasis(n_sites=2)


def twosite_hamop(basis):
    (c1u, c2u), (c1d, c2d) = basis.annihilation_ops()
    u_op = (c1u.dag * c1u * c1d.dag * c1d)
    mu_op = (c1u.dag * c1u) + (c1d.dag * c1d)
    eps_op = (c2u.dag * c2u) + (c2u.dag * c2u)
    v_op = (c1u.dag * c2u) + (c2u.dag * c1u) + (c1d.dag * c2d) + (c2d.dag * c1d)
    return HamiltonOperator(u=u_op, mu=-mu_op, eps=eps_op, v=v_op)

def twosite_hamop_sigma(u=4, eps=2, mu=2, v=1):
    u_op = 1/2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    mu_op = (kron(sz, si, si, si) + kron(si, si, sz, si))
    eps_op = (kron(si, sz, si, si) + kron(si, si, si, sz))
    v_op = kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(sy, sy, sx, sx) + kron(si, si, sy, sy)
    return 1/2 * (u * u_op + mu * mu_op - eps * eps_op + v * v_op)
