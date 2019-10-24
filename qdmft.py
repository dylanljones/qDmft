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
from qsim import Circuit
from qsim.vqe import VqeSolver

VQE_FILE = "circuits/twosite_vqe"


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

    c = Circuit.load(VQE_FILE)
    c.add_qubit(0)

    c.append(time_evolution_circuit(arg, 1))
    c.m(0, 0)
    c.print(False)
    res = c.run(100, verbose=True)
    print(c.backend.last)
    res.show_histogram()
    # s.apply_gate(xy_gate(np.pi/3))


if __name__ == "__main__":
    main()
