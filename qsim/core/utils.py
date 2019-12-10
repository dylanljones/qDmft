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
    """ Computes the Kronecker product of two or more arrays.

    Parameters
    ----------
    args: list of array_like or array_like
        Arrays used for the kronecker product.

    Returns
    -------
    out: np.ndarray
    """
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def is_unitary(a, rtol=1e-5, atol=1e-8):
    """ Checks if an array is a unitary matrix.

    Parameters
    ----------
    a: array_like
        The array to check.
    rtol: float
        The relative tolerance parameter.
    atol: float
        The absolute tolerance parameter.

    Returns
    -------
    allclose: bool
    """
    a = np.asarray(a)
    return np.allclose(a.dot(np.conj(a).T), np.eye(a.shape[0]), rtol=rtol, atol=atol)


def get_projector(v):
    r""" Constructs a projection-matrix from a given vector.

    .. math::
        P = |v><v|

    Parameters
    ----------
    v: (N) np.ndarray
        The state-vector used for constructing the projector.

    Returns
    -------
    p: (N, N) np.ndarray
    """
    if len(v.shape) == 1:
        v = v[:, np.newaxis]
    return np.dot(v, np.conj(v).T)


def expectation(o, psi):
    r""" Computes the expectation value of an operator in a given state.

    .. math::
        x = <\Psi| \hat{O} |\Psi>

    Parameters
    ----------
    o: (N, N) np.ndarray
        Operator in matrix representation.
    psi: (N) np.ndarray
        State in vector representation.

    Returns
    -------
    x: float
    """
    return np.dot(np.conj(psi).T, o.dot(psi)).real


def binstr(x, n=None):
    """ Constructs the string of a binary number.

    Parameters
    ----------
    x: int
        Integer of binary value.
    n: int, optional
        Length of the binary string.

    Returns
    -------
    s: str
    """
    string = bin(x)[2:]
    n = n or len(string)
    return f"{string:0>{n}}"


def basis_states(n):
    """ Constructs basis states in binary representation.

    Parameters
    ----------
    n: int
        Number of states.

    Returns
    -------
    states: list of int
    """
    return list(range(int(n)))


def basis_strings(n):
    """ Constructs strings of basis states.

    Parameters
    ----------
    n: int
        Number of states.

    Returns
    -------
    state_strings: list of str
    """
    return [f"|{binstr(x, n)}>" for x in range(2 ** n)]


# =========================================================================


def to_array(x, *args, **kwargs):
    """ Ensures argument is returned as a numpy array

    Parameters
    ----------
    x: int or float or list or np.ndarray
        Argument to convert to array.
    args: list
        Optional positional arguments for array creation.
    kwargs: list
        Optional keyword arguments for array creation.

    Returns
    -------
    a: np.ndarray
    """
    if not hasattr(x, "__len__"):
        x = [x]
    return np.asarray(x, *args, **kwargs)


def to_list(x):
    """ Ensures argument is returned as a list.

    Parameters
    ----------
    x: int or float or list or np.ndarray
        Argument to convert to list.

    Returns
    -------
    l: list
    """
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
    """ Extracts a keyword-value from a string

    Parameters
    ----------
    string: str
        Full string containing keyword-values.
    key: str
        The keyword follwed by the desired value.
    delim: str, optional
        Delimiter used between the values.

    Returns
    -------
    value: str
    """
    pre = key + "="
    if not string.endswith(delim):
        string += delim
    res = re.search(pre + r'(.*?)' + delim, string)
    return res.group(1) if res is not None else ""


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


def binary_histogram(data, normalize=True):
    n, n_bins = data.shape
    binvals = np.power(2, np.arange(n_bins))[::-1]
    data = np.sum(data * binvals[np.newaxis, :], axis=1)
    hist, edges = np.histogram(data, bins=np.arange(2 ** n_bins + 1))
    bins = edges[:-1].astype("int")  # + 0.5
    if normalize:
        hist = hist / n
    return bins, hist


def plot_binary_histogram(bins, hist, labels=None, padding=0.2, color=None, alpha=0.9, max_line=True, lc="r", lw=1):
    plot = Plot(xlim=(-0.5, len(bins) - 0.5), ylim=(0, 1.1))
    plot.grid(axis="y")
    plot.set_ticks(bins, np.arange(0, 1.1, 0.2))
    if labels is not None:
        plot.set_ticklabels(labels)
    plot.bar(bins, hist, width=1-padding, color=color, alpha=alpha)
    if max_line:
        ymax = np.max(hist)
        plot.draw_lines(y=ymax, color=lc, lw=lw)
    return plot
