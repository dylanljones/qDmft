# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from qsim import *
from scitools import Plot


def tb_hamiltonian(n, w=1, t=1):
    ham = np.zeros((n, n), dtype="float")
    np.fill_diagonal(ham, w * np.random.uniform(0.0, 1.0, size=n))
    for i in range(n-1):
        j = i + 1
        ham[i, j] = ham[j, i] = t
    return ham
    

def main():
    print(tb_hamiltonian(4, w=0.1))

if __name__ == "__main__":
    main()
