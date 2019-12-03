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
from qsim.twosite import gf_fit, gf_spectral

si, sx, sy, sz = pauli
omega_1 = 4.033
omega_2 = 5.197
alpha_1 = 0.242
alpha_2 = 0.207


def test_fft():
    c = Circuit(5)


def main():
    test_fft()


if __name__ == "__main__":
    main()
