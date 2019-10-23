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
