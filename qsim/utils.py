# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np


def kron(operators):
    x = np.array([[1]])
    for op in operators:
        x = np.kron(x, op)
    return x
