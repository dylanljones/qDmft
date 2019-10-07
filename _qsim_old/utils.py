# -*- coding: utf-8 -*-
"""
Created on 22 Sep 2019
author: Dylan Jones

project: Qsim
version: 1.0
"""
import numpy as np


def kron(operators):
    x = np.array([[1]])
    for op in operators:
        x = np.kron(x, op)
    return x


class Bit(int):

    def __new__(cls, i, register, value=0):
        self = int.__new__(cls, i)
        self.val = value
        self.reg = register
        return self


class Register:

    bit_type = Bit

    def __init__(self, n):
        self.n = n
        self.bits = [self.bit_type(i, self) for i in range(n)]

    def __str__(self):
        return f"Register({self.bits})"


class QBit(Bit):

    def __new__(cls, value, register):
        return super().__new__(cls, value, register)
