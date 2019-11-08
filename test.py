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
import itertools
from scitools import Matrix
from qsim import *
from qsim import Result

si, sx, sy, sz = pauli


def test_rotation(c):
    n = 100
    phi = np.linspace(0, 2*np.pi, n)
    values = np.zeros(n)
    for i in range(n):
        c.set_param(0, phi[i])
        res = c.run(100)
        values[i] = res.mean().real
    plot = Plot()
    plot.plot(phi, values)
    plot.show()


def main():
    c = Circuit(2, 1)
    c.rx(0)
    c.mz(0)
    c.print()


    test_rotation(c)


if __name__ == "__main__":
    main()
