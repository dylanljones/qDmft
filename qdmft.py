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
from qsim2 import kron, pauli, single_gate, cgate, X_GATE, rx_gate
from qsim2 import Circuit
from qsim2.vqe import VqeSolver

si, sx, sy, sz = pauli

FILE = "circuits/twosite_vqe"


def hamiltonian(u=4, v=1, eps_bath=2, mu=2):
    h1 = u / 2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    h2 = mu * (kron(sz, si, si, si) + kron(si, si, sz, si))
    h3 = - eps_bath * (kron(si, sz, si, si) + kron(si, si, si, sz))
    h4 = v * (kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy))
    return 1/2 * (h1 + h2 + h3 + h4)


def vqe_circuit(depth=2):
    c = Circuit(4)
    c.h(0)
    for i in range(depth):
        c.cx(0, 1)
        c.cx(0, 2)
        c.cx(0, 3)
        c.ry(1)
        c.ry(2)
        c.ry(3)
    c.h(0)
    return c


def optimize_circuit(new=False, depth=2, file=FILE):
    print()
    if not new:
        try:
            c = Circuit.load(file)
            print(f"Circuit \"{file}\" loaded!")
            return c
        except FileNotFoundError:
            print(f"No file {file} found.")
    vqe = VqeSolver(hamiltonian(), vqe_circuit(depth))
    vqe.solve(verbose=True)
    file = vqe.save(file)
    print(f"Saving circuit: \"{file}\"")
    print()
    return vqe.circuit


def xy_gate(arg):
    notc = cgate(1, 0, X_GATE)
    crx = cgate(0, 1, rx_gate(arg))
    xy = np.dot(notc, crx.dot(notc))
    return xy


def add_xy_gate(c, q0, q1, arg):
    c.cx(q1, q0)
    c.crx(q0, q1, arg)
    c.cx(q1, q0)


def trotter_step(c, arg):
    add_xy_gate(c, 0, 1, arg)
    add_xy_gate(c, 2, 3, arg)





def main():
    # c = optimize_circuit(True)
    # c.add_qubit(0, add_clbit=True)
    arg = np.pi/3
    c = Circuit(4)
    trotter_step(c, arg)
    c.print()
    c.run()
    # s.apply_gate(xy_gate(np.pi/3))
    print(c.backend)



if __name__ == "__main__":
    main()
