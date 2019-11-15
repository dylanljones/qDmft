# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from scitools import Plot
from qsim import kron, pauli
from qsim import Circuit, VqeSolver, prepare_ground_state, test_vqe

VQE_FILE = "circuits/twosite_vqe"

si, sx, sy, sz = pauli

# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def hamiltonian(u=4, v=1, eps_bath=2, mu=2):
    h1 = u / 2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    h2 = mu * (kron(sz, si, si, si) + kron(si, si, sz, si))
    h3 = - eps_bath * (kron(si, sz, si, si) + kron(si, si, si, sz))
    h4 = v * (kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy))
    return 1/2 * (h1 + h2 + h3 + h4)


def config_vqe_circuit(vqe):
    c = vqe.circuit
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def prepare_ground_state2(new=False, file=VQE_FILE, verbose=True):
    print()
    if not new:
        try:
            c = Circuit.load(file)
            if verbose:
                print(f"Circuit: {file} loaded!")
            return c
        except FileNotFoundError:
            print(f"No file {file} found.")
    vqe = VqeSolver(hamiltonian(), 1)
    config_vqe_circuit(vqe)
    vqe.solve(verbose=verbose)
    file = vqe.save(file)
    if verbose:
        print(f"Saving circuit: {file}")
        print()
    return vqe.circuit


def test_gs_preparation(circuit):
    test_vqe(circuit, hamiltonian())


# =========================================================================
#                            TIME EVOLUTION
# =========================================================================


def time_evolution_circuit(arg, step):
    c = Circuit(5, 1)
    c.h(0)
    c.cx(0, 1)
    for i in range(step):
        c.xy(1, 2, arg)
        c.xy(3, 4, arg)
        c.b(1, 3, arg)
    c.cy(0, 1)
    c.h(0)
    return c


def get_twosite_circuit(arg, step):
    c = prepare_ground_state()
    c.add_qubit(0)
    c.append(time_evolution_circuit(arg, step))
    return c


def main():
    tau = 6
    v = 4
    n = 20
    arg = v/2 * tau/n
    ham = hamiltonian()
    c = prepare_ground_state(ham, config_vqe_circuit, file=VQE_FILE)
    c.print()
    # s.apply_gate(xy_gate(np.pi/3))


if __name__ == "__main__":
    main()
