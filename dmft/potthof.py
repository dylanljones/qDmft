# -*- coding: utf-8 -*-
"""
Created on 14 Feb 2019
author: Dylan

Reference functions taken from M. Potthof: 'Two-site dynamical mean-field theory'

project: qDmft
version: 1.0
"""
import numpy as np
from itertools import product
from scipy import linalg as la


def gf_imp0(z, eps0, eps1, t):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + t*t)
    term1 = (r - e) / (z - e - r)
    term2 = (r + e) / (z - e + r)
    return 1/(2*r) * (term1 + term2)
