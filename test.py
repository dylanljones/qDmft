# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import itertools
from qsim import *

si, sx, sy, sz = pauli


arg = kron(sx, sx)
print(arg.real)
print(arg.real * arg.real)
print(arg.real * arg.real * arg.real)


arg = kron(sy, sy)
print(arg.real)
print(arg.real * arg.real)
print(arg.real * arg.real * arg.real)


def main():
    reg = QuRegister(2)
    s = StateVector(reg)





if __name__ == "__main__":
    main()
