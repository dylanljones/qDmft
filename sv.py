# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
from qsim2 import Statevector
from qsim2.gates import HADAMARD_GATE, X_GATE



def main():
    print()
    s = Statevector("000")
    # s.apply_gate(HADAMARD_GATE, 0)
    print(s)
    s.apply_gate(X_GATE, 0)
    s.apply_gate(HADAMARD_GATE, 0)
    print(s)


    print(s.probabilities())




if __name__ == "__main__":
    main()
