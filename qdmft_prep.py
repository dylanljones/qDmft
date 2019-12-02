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
from scitools import Plot, Matrix
from qsim import kron, pauli
from qsim import Circuit, VqeSolver, prepare_ground_state, test_vqe
from qsim.fermions import HamiltonOperator
VQE_FILE = "circuits/twosite_vqe"

si, sx, sy, sz = pauli

X0 = [5.7578, 3.1416, 3.6519, 2.8804, 4.5688, 5.9859, 4.187, 4.7124]

# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================

def hamiltonian(u=4, eps=2, mu=2, v=1):
    u_op = 1/2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    mu_op = (kron(sz, si, si, si) + kron(si, si, sz, si))
    eps_op = (kron(si, sz, si, si) + kron(si, si, si, sz))
    v_op = kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy)
    return 1/2 * (u * u_op + mu * mu_op - eps * eps_op + v * v_op).real


def config_vqe_circuit(c):
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def run_vqe(ham):
    n_qubits = int(np.log2(ham.shape[0]))
    c = Circuit(n_qubits, 0)
    config_vqe_circuit(c)
    c.print()

    def func(theta):
        c.set_params(theta)
        c.run_shot()
        return c.expectation(ham)

    x0 = np.random.uniform(0, 2*np.pi, c.n_params)
    sol = optimize.minimize(func, x0=x0, method="Nelder-Mead")
    print(sol)


def main():
    ham = Matrix(hamiltonian())
    run_vqe(ham)

    # s.apply_gate(xy_gate(np.pi/3))


if __name__ == "__main__":
    main()
