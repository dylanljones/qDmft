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
from scitools import Plot, Colors


def main():
    c = Circuit(2)
    c.h(0)
    c.crx(0, 1, np.pi/3)
    c.m()
    c.run(1000, verbose=True)

    print(c.backend.last)
    c.show_histogram(color=Colors.bblue, lc=Colors.bred)


if __name__ == "__main__":
    main()
