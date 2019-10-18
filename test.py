# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import re, os
import numpy as np
from qsim import Circuit, Gate
from qsim.visuals import CircuitString
from qsim.utils import *
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


def main():
    # c = get_circuit(True)
    # print(c)
    # c.print()
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    # c.m()
    print(c)
    c.add_qubit(0)
    print(c)

    c.m()

    c.print()
    res = c.run(100)
    print(res)
    res.show_histogram()



if __name__ == "__main__":
    main()
