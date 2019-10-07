# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: utils$
version: 1.0
"""
import numpy as np


def kron(operators):
    x = np.array([[1]])
    for op in operators:
        x = np.kron(x, op)
    return x
