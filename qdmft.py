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
from qsim2 import kron, pauli
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
    c = Circuit(4, 1)
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


def time_evolution_circuit(arg, step):
    c = Circuit(5, 1)
    c.h(0)
    c.cx(0, 1)
    for i in range(step):
        c.xy(1, 2, arg)
        c.xy(3, 4, arg)
        c.b(1, 3, arg)
    c.cx(0, 1)
    c.h(0)
    return c


def main():
    tau = 6
    v = 4
    n = 2
    arg = v/2 * tau/n

    # c = Circuit(5, 1)
    c = optimize_circuit()
    c.add_qubit(0)

    c.append(time_evolution_circuit(arg, 1))
    c.m(0, 0)
    # c.print(False)
    c.run(1000, verbose=True)
    c.res.show_histogram()
    # s.apply_gate(xy_gate(np.pi/3))


if __name__ == "__main__":
    main()
