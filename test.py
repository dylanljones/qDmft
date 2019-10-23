# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
from qsim import QuRegister, Qubit, Circuit, Gate, kron
from qsim.core.gates import X_GATE, cgate, rx_gate, single_gate, pauli
from scitools import Plot


def get_circuit(new=False, file="circuits/test.circ"):
    if new or not os.path.isfile(file):
        print(f"Saving circuit: {file}")
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        # c.m()
        c.save("circuits/test")
        return c
    else:
        print(f"Loading circuit: {file}")
        return Circuit.load(file)


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
    c.m(0, 0)
    return c


def main():
    v, t, n = 4, 6, 12
    arg = v/2 * t/n

    c = Circuit(2)
    c.h(0)
    c.h(0)
    c.m()
    c.run(100)
    c.show_histogram()


if __name__ == "__main__":
    main()
