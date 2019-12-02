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
from scitools import Plot
from qsim.core import *
from qsim import Circuit, Gate
from qsim.twosite import gf_fit

si, sx, sy, sz = pauli


def main():
    d = d_gatefunc(0, 2, 1)
    print(d)


if __name__ == "__main__":
    main()
