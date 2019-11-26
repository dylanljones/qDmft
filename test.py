# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from scipy.linalg import expm
import scipy.linalg as la
from qsim.core import *
from qsim import Circuit, Gate

si, sx, sy, sz = pauli


def main():
    c = Circuit(2, 2)
    c.my()
    res = c.run(1000, state0=kron(ONE, ZERO))
    print(np.mean(res, axis=0))


if __name__ == "__main__":
    main()
