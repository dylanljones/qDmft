# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
from itertools import product, permutations
from qsim import kron, Statevector, ONE, ZERO
from qsim.gates import HADAMARD_GATE

ZERO2 = np.array([1, 0])


def main():
    s = kron(ZERO, ZERO)
    h = kron(np.eye(2), HADAMARD_GATE)
    print(np.dot(h, s))
    print(s)
    print(h)


if __name__ == "__main__":
    main()
