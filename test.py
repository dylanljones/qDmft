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


def main():
    c = Circuit(2, 1)
    c.h(0)
    c.mz(0)
    res = c.run()
    print(res.data)
    print(res)



if __name__ == "__main__":
    main()
