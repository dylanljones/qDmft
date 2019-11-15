# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import scipy.linalg as la
from scipy import sparse
import itertools
from scitools import Matrix
from qsim.core import *
from qsim import Circuit, Gate, prepare_ground_state, test_vqe
from qsim.fermions import FBasis, Operator, HamiltonOperator

si, sx, sy, sz = pauli


def transform_ops(c1u, c2u, c1d, c2d):
    c1d, c2d = c1d.jordan_wigner(2, 0), c2d.jordan_wigner(2, 1)
    c1u, c2u = c1u.jordan_wigner(2, 2), c2u.jordan_wigner(2, 3)
    return c1u, c2u, c1d, c2d


def siam_hamop(c1u, c2u, c1d, c2d):
    u_op = c1u.dag * c1u * c1d.dag * c1d
    mu_op = (c1u.dag * c1u) + (c1d.dag * c1d)
    eps_op = (c2u.dag * c2u) + (c2d.dag * c2d)
    v_op = (c1u.dag * c2u) + (c2u.dag * c1u) + (c1d.dag * c2d) + (c2d.dag * c1d)
    return HamiltonOperator(u=u_op, mu=mu_op, eps=eps_op, v=v_op)


def ground_energy(ham):
    eigvals, eigvecs = la.eig(ham.todense())
    return np.min(eigvals)


def hamiltonian(u=4, v=1, eps_bath=2, mu=2):
    h1 = u / 2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    h2 = mu * (kron(sz, si, si, si) + kron(si, si, sz, si))
    h3 = - eps_bath * (kron(si, sz, si, si) + kron(si, si, si, sz))
    h4 = v * (kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy))
    return 1/2 * (h1 + h2 + h3 + h4)


# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def config_vqe_circuit(vqe):
    c = vqe.circuit
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def test_new_vqe():
    u, v = 4, 1
    mu = eps = u/2
    basis = FBasis(2)
    (c1u, c2u), (c1d, c2d) = basis.annihilation_ops()
    c1u, c2u, c1d, c2d = transform_ops(c1u, c2u, c1d, c2d)
    hamop = siam_hamop(c1u, c2u, c1d, c2d)
    ham = hamop.build(u=u, mu=-mu, eps=eps, v=-v).todense()

    ham = hamiltonian()
    prepare_ground_state(ham, config_vqe_circuit, file="test.circ", verbose=True)


def main():
    # c = Circuit(2)
    # c.x(0)
    # c.ry([0, 1], [1, 2])
    # c.save("Test2")
    c = Circuit.load("Test2")
    c.print()


if __name__ == "__main__":
    main()
