# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import scipy.linalg as la
from scipy import sparse
from qsim.core import *
from qsim import Circuit, Gate, prepare_ground_state, test_vqe
from qsim.fermions import FBasis, Operator, HamiltonOperator

si, sx, sy, sz = pauli


def main():
    c = Circuit(4)
    c.ry([0, 1], [1, 2])
    c.run()
    c.print()

if __name__ == "__main__":
    main()
