# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from scipy import optimize
from scitools import Matrix, Plot
from qsim import kron, pauli
from qsim import Circuit, VqeSolver, prepare_ground_state, test_vqe
from qsim.fermions import HamiltonOperator
VQE_FILE = "circuits/twosite_free_vqe"

si, sx, sy, sz = pauli


# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def hamiltonian(eps=0, v=1):
    eps_op = -eps/2 * (kron(si, sz, si, si) + kron(si, si, si, sz))
    v_op = v/2 *(kron(sx, sx, si, si) + kron(sy, sy, si, si) +
                 kron(si, si, sx, sx) + kron(si, si, sy, sy))
    return Matrix((eps_op + v_op).real)


def config_vqe_circuit2(c):
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def config_vqe_circuit(c):
    c.ry(0)
    c.ry(1)
    c.ry(2)
    c.ry(3)
    c.cx(2, 3)

    c.ry(2)
    c.ry(3)
    c.cx(0, 2)

    c.ry(0)
    c.ry(2)
    c.cx(0, 1)
    return c


def get_groundstate(ham):
    eigvals, _ = np.linalg.eig(ham)
    print(eigvals)
    return np.min(eigvals)



def main():
    ham = hamiltonian()
    # ham.show()
    vqe = VqeSolver(ham)
    config_vqe_circuit(vqe.circuit)

    sol = vqe.solve(method="Nelder-Mead", options={"maxiter": 1000})
    print(sol)



if __name__ == "__main__":
    main()
