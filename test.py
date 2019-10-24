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
from qsim import QuRegister, Qubit, StateVector, Circuit, Gate, kron, pauli
from qsim.core.gates import X_GATE, cgate, rx_gate, single_gate
from scitools import Plot

si, sx, sy, sz = pauli


def main():
    reg = QuRegister(1)
    x = Gate.x(reg[0])
    z = Gate.z(reg[0])

    s = StateVector(reg.bits)
    # s.apply_gate(x)
    s.apply_gate(z)
    print(s)
    s.apply_gate(z)
    proj = sz
    print(np.dot(s.amp, proj))



if __name__ == "__main__":
    main()
