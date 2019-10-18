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


def histogram(data, normalize=True):
    n, n_bins = data.shape
    binvals = np.power(2, np.arange(n_bins))[::-1]
    data = np.sum(data * binvals[np.newaxis, :], axis=1)
    hist, edges = np.histogram(data, bins=np.arange(2 ** n_bins+1))
    bins = edges[:-1]  # + 0.5
    if normalize:
        hist = hist / n
    return bins, hist


class Result:

    def __init__(self, data):
        self.data = None
        self.hist = None
        self.load(data)

    def load(self, data):
        self.data = data
        self.hist = np.asarray(histogram(data, normalize=True))

    @property
    def shape(self):
        return self.data.shape

    @property
    def n(self):
        return self.shape[0]

    def sorted(self):
        probs = self.hist[1]
        indices = np.argsort(probs)[::-1]
        return np.asarray([indices, probs[indices]]).T

    def show_histogram(self, show=True):
        bins, hist = self.hist
        plot = Plot()
        plot.ax.bar(bins, hist, width=0.9)
        plot.grid(which="y")
        if show:
            plot.show()


def main():
    c = get_circuit()
    c.print()
    res = Result(c.run(100))
    print(f"Results after {res.n} runs:")
    for x, p in res.sorted():
        print(f"{x}; {p}")

    res.show_histogram()

    # c.show_histogram()
    # print(c.backend.snapshots[-1])
    # print(c.backend)


if __name__ == "__main__":
    main()
