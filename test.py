# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
from qsim import Circuit, Gate
from qsim.visuals import CircuitString
from qsim.utils import *
from scitools import Plot


def get_circuit(new=False, file="circuits/test.circ"):
    if new or not os.path.isfile(file):
        print(f"Saving circuit: {file}")
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        # c.m()
        c.save("circuits/test")
        return c
    else:
        print(f"Loading circuit: {file}")
        return Circuit.load(file)


class Register(list):

    INDEX = 0

    def __init__(self, size):
        self.idx = Register.INDEX
        Register.INDEX += 1
        super().__init__(range(size))

    @property
    def n(self):
        return len(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(idx: {self.idx}, size: {self.n})"


class QuRegister(Register):

    def __init__(self, size):
        super().__init__(size)


class ClRegister(Register):

    def __init__(self, size, values=0):
        super().__init__(size)
        if not hasattr(values, "__len__"):
            values = np.ones(size) * values
        self.values = np.asarray(values)

    @classmethod
    def like(cls, register):
        return cls(register.n)

    def __str__(self):
        return f"{self.__class__.__name__}(idx: {self.idx}, values: {self.values})"


def main():
    qreg = QuRegister(3)
    creg = ClRegister.like(qreg)
    print(qreg)
    print(creg)


if __name__ == "__main__":
    main()
