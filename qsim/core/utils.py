# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re
import numpy as np
from scitools import Plot

si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = si, sx, sy, sz

EIGVALS = np.array([+1, -1])
EV_X = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
EV_Y = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
EV_Z = np.array([[1, 0], [0, 1]])

# Initial states
ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)
IPLUS = np.array([1, 1j]) / np.sqrt(2)
IMINUS = np.array([1, -1j]) / np.sqrt(2)

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


def get_projector(v):
    if len(v.shape) == 1:
        v = v[:, np.newaxis]
    return np.dot(v, np.conj(v).T)


def expectation(op, v):
    return np.dot(np.conj(v).T, op.dot(v))


def binstr(x, n=None):
    string = bin(x)[2:]
    n = n or len(string)
    return f"{string:0>{n}}"


def basis_states(n):
    return list(range(int(n)))


def basis_strings(n):
    return [f"|{binstr(x, n)}>" for x in range(2 ** n)]


# =========================================================================


def to_array(x, *args, **kwargs):
    if not hasattr(x, "__len__"):
        x = [x]
    return np.asarray(x, *args, **kwargs)


def to_list(x):
    if not hasattr(x, "__len__"):
        x = [x]
    return list(x)


def str_to_list(s, dtype=int):
    """ Convert a string of numbers into list of given data-type

    Parameters
    ----------
    s: str
        String of list
    dtype: type
        Data type of the list

    Returns
    -------
    data_list: list
    """
    s = s.strip()
    if s == "None":
        return None
    pattern = r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?"
    return [dtype(x) for x in re.findall(pattern, s)]


def get_info(string, key, delim=";"):
    pre = key + "="
    if not string.endswith(delim):
        string += delim
    res = re.search(pre + r'(.*?)' + delim, string)
    return res.group(1) if res is not None else ""


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


class Result:

    def __init__(self, data):
        self.data = None
        self.basis = None
        self.hist = None
        self.load(data)

    def load(self, data, normalize=True):
        self.data = data
        self.basis = Basis(data.shape[1])
        self.hist = histogram(data, normalize)

    @property
    def shape(self):
        """tuple: shape of the data array (n_measurements, n_bits)"""
        return self.data.shape

    @property
    def n(self):
        """int: number of measurments"""
        return self.shape[0]

    @property
    def n_bits(self):
        """int: number of bits"""
        return self.shape[1]

    @property
    def labels(self):
        """list of str: Labels of the basis-states of the measured qubits"""
        return self.basis.labels

    def sorted(self):
        """ Returns the sorted bins of the measurment histogram data (descending probability)

        Returns
        -------
        hist: array_like
            sorted histogram data with descending probability
        """
        bins, probs = self.hist
        indices = np.argsort(probs)[::-1]
        return list([(bins[i], probs[i]) for i in indices])

    def expected(self):
        """ Returns the most occuring binary value and the corresponding simulate_probability

        Returns
        -------

        value: int
            Expected binary result (0 or 1)
        p: float
            Probability of the result
        """
        return self.sorted()[0]

    def mean(self):
        """ returns the mean of the measurments

        mean: np.ndarray
            mean value of each qubit for all measurments
        """
        return np.mean(self.data, axis=0)

    def highest(self, thresh=0.7):
        res_sorted = self.sorted()
        pmax = res_sorted[0][1]
        return [(self.labels[i], p) for i, p in res_sorted if p >= thresh * pmax]

    def show_histogram(self, show=True, print_values=True, max_line=True, padding=0.2,
                       color=None, alpha=0.9, lc="r", lw=1, text_padding=0):
        bins, hist = self.hist
        plot = Plot(xlim=(-0.5, len(bins) - 0.5), ylim=(0, 1.1), title=f"N={self.n}")
        plot.grid(axis="y")
        plot.set_ticks(bins, np.arange(0, 1.1, 0.2))
        plot.set_ticklabels(self.labels)
        # plot.draw_lines(y=1, color="0.5")
        plot.bar(bins, hist, width=1-padding, color=color, alpha=alpha)
        ymax = np.max(hist)
        if print_values:
            ypos = ymax + text_padding + 0.02
            for x, y in zip(bins, hist):
                col = "0.5" if y != ymax else "0.0"
                if y:
                    plot.text((x, ypos), s=f"{y:.2f}", ha="center", va="center", color=col)
        if max_line:
            plot.draw_lines(y=ymax, color=lc, lw=lw)

        if show:
            plot.show()
        return plot

    def __str__(self):
        entries = [f"   {label} {p:.3f}" for label, p in self.highest()]
        string = f"Result ({self.n} shots):\n"
        string += "\n".join(entries)
        return string
