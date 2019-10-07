# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: states$
version: 1.0
"""
import numpy as np

ZERO = np.array([[1, 0]]).T
ONE = np.array([[0, 1]]).T
PLUS = np.array([[1, 1]]).T / np.sqrt(2)
MINUS = np.array([[1, -1]]).T / np.sqrt(2)
P0 = np.dot(ZERO, ZERO.T)
P1 = np.dot(ONE, ONE.T)