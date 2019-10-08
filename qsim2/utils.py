# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
from scitools import Plot, Circle

# Initial states
ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)
IPLUS = np.array([1, 1j]) / np.sqrt(2)
IMINUS = np.array([1, 1j]) / np.sqrt(2)

#State dictionary for easy initialization
STATES = {"0": ZERO, "1": ONE, "+": PLUS, "-": MINUS, "i+": IPLUS, "i-": IMINUS}

# Projections onto |0> and |1>
P0 = np.dot(ZERO[:, np.newaxis], ZERO[np.newaxis, :])
P1 = np.dot(ONE[:, np.newaxis], ONE[np.newaxis, :])


def kron(*args):
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def basis_states(n):
    return list(range(int(n)))


def binstr(x, n):
    return f"{bin(x)[2:]:0>{n}}"
