# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import re, os
import numpy as np
from qsim import Circuit, Gate
from qsim.visuals import CircuitString
from qsim.utils import *
from scitools import Plot


def get_circuit(file="circuits/test.circ", new=False):
    if new or not os.path.isfile(file):
        print(f"Saving circuit: {file}")
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.m()
        c.save("circuits/test")
        return c
    else:
        print(f"Loading circuit: {file}")
        return Circuit.load(file)


class Result(np.ndarray):

    def __new__(cls, inputarr, dtype=None):
        obj = np.asarray(inputarr, dtype).view(cls)
        return obj

    @property
    def n(self):
        return self.shape[0]

    def hist(self):
        n, nbits = self.shape
        binvals = np.power(2, np.arange(nbits))[::-1]
        data = np.sum(self * binvals[np.newaxis, :], axis=1)
        hist, edges = np.histogram(data, bins=np.arange(2 ** nbits+1))
        bins = edges[:-1] + 0.5
        return bins, hist / n


def main():
    c = get_circuit()

    res = Result(c.run(100))
    print(res.hist())


    c.show_histogram()
    # print(c.backend.snapshots[-1])
    # print(c.backend)


if __name__ == "__main__":
    main()
