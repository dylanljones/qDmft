# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import re
import numpy as np
from scitools import Plot, Circle

# Initial states
ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)
IPLUS = np.array([1, 1j]) / np.sqrt(2)
IMINUS = np.array([1, 1j]) / np.sqrt(2)

# State dictionary for easy initialization
STATES = {"0": ZERO, "1": ONE, "+": PLUS, "-": MINUS, "i+": IPLUS, "i-": IMINUS}

# Projections onto |0> and |1>
P0 = np.dot(ZERO[:, np.newaxis], ZERO[np.newaxis, :])
P1 = np.dot(ONE[:, np.newaxis], ONE[np.newaxis, :])
PROJECTIONS = [P0, P1]


def kron(*args):
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def to_array(x, *args, **kwargs):
    if not hasattr(x, "__len__"):
        x = [x]
    return np.asarray(x, *args, **kwargs)


def to_list(x):
    if not hasattr(x, "__len__"):
        x = [x]
    return list(x)


def str_to_list(string, dtype=int):
    if string.strip() == "None":
        return None
    string = string.replace("[").replace("]")
    return [dtype(x) for x in string.split(" ")]


def basis_states(n):
    return list(range(int(n)))


def binstr(x, n=None):
    string = bin(x)[2:]
    n = n or len(string)
    return f"{string:0>{n}}"


def basis_strings(n):
    return [f"|{binstr(x, n)}>" for x in range(2 ** n)]


def get_info(string, key, delim="; "):
    pre = key + "="
    return re.search(pre + r'(.*?)' + delim, string).group(1)


def histogram(data, normalize=True):
    n, n_bins = data.shape
    binvals = np.power(2, np.arange(n_bins))[::-1]
    data = np.sum(data * binvals[np.newaxis, :], axis=1)
    hist, edges = np.histogram(data, bins=np.arange(2 ** n_bins+1))
    bins = edges[:-1].astype("int")  # + 0.5
    if normalize:
        hist = hist / n
    return bins, hist


class Basis:

    def __init__(self, n):
        self.qbits = n
        self.n = 2 ** n
        self.states = basis_states(self.n)
        self.labels = [f"|{binstr(x, n)}>" for x in range(self.n)]

    def get_indices(self, qubit, val):
        idx = self.qbits - qubit - 1
        return [i for i in self.states if (i >> idx & 1) == val]

    def __getitem__(self, item):
        return self.labels[item]

    def __str__(self):
        return "Basis(" + ", ".join(self.labels) + ")"
