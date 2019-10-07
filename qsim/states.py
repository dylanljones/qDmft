# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np

ZERO = np.array([[1, 0]]).T
ONE = np.array([[0, 1]]).T
PLUS = np.array([[1, 1]]).T / np.sqrt(2)
MINUS = np.array([[1, -1]]).T / np.sqrt(2)
P0 = np.dot(ZERO, ZERO.T)
P1 = np.dot(ONE, ONE.T)

STATES = {"0": ZERO, "1": ONE, "+": PLUS, "-": MINUS}
