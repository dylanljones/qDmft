# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import itertools
from qsim import QuRegister, Qubit, StateVector, Circuit, Gate, kron
from qsim.core.gates import X_GATE, cgate, rx_gate, single_gate, pauli
from scitools import Plot


def main():
    reg = QuRegister(2)

    x = Gate.x(reg[0])

    s = StateVector(reg.bits)
    s.apply_gate(x)
    print(s)


if __name__ == "__main__":
    main()
