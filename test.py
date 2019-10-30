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

si, sx, sy, sz = pauli


def time_evolution_operator(ham, t):
    return np.exp(-1j * ham * t)







def main():
    ham = np.zeros((3, 3))
    ham[2, 1] = 1
    ham[1, 2] = 1
    print(ham)
    eigvals, eigvecs = la.eig(ham)
    eigvecs = eigvecs.T
    for i in range(3):
        ev, v = eigvals[i], eigvecs[i]
        print(np.all(np.dot(ham, v) == ev * v))
        print(f"{ev.real:<4} {v}")










if __name__ == "__main__":
    main()
