# -*- coding: utf-8 -*-
"""
Created on 18 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy.sparse import csr_matrix

si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
s_minus = (sx - 1j * sy).real


def kron(*args):
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def _ordering_phase(spins, i, spin, fermions=True):
    if not fermions:
        return 1
    state = spins[spin]
    particles = bin(state >> i + 1)[2:].count("1")
    return 1 if particles % 2 == 0 else -1


def annihilation_op_csr(basis, idx, spin):
    n = basis.n
    row, col, data = list(), list(), list()
    for j, state in enumerate(basis):
        other = state.annihilate(idx, spin)
        if other is not None:
            try:
                i = basis.index(other)
                val = _ordering_phase(state.spins, idx, spin, fermions=True)
                row.append(i), col.append(j), data.append(val)
            except ValueError:
                pass
    return csr_matrix((data, (row, col)), shape=(n, n), dtype="int")


class Operator:

    CHARS = "↑", "↓"

    def __init__(self, array=None):
        if isinstance(array, Operator):
            array = array.csr
        self.csr = csr_matrix(array)

    @classmethod
    def annihilation_operator(cls, idx, spin, basis):
        mat = annihilation_op_csr(basis, idx, spin)
        return cls(idx, spin, mat)

    @classmethod
    def creation_operator(cls, idx, states, spin):
        return cls.annihilation_operator(idx, states, spin).dag

    def todense(self):
        return self.csr.todense()

    def jordan_wigner(self, idx, spin, n_sites):
        i = 2 * idx
        idx = i - 1 if spin == 0 else i
        parts = [sz] * idx + [si] * (2 * n_sites - idx)
        parts[idx] = s_minus
        return 0.5 * Operator(np.conj(kron(parts)).T)

    @property
    def dense(self):
        return self.csr.todense()

    @property
    def T(self):
        return self.like(self.csr.T)

    @property
    def dag(self):
        return Operator(np.conj(self.csr).T)

    @property
    def abs(self):
        return Operator(np.abs(self.csr))

    @property
    def nop(self):
        return Operator(self.dag * self)

    @staticmethod
    def _get_value(other):
        if isinstance(other, Operator):
            other = other.csr
        return other

    def dot(self, other):
        return Operator(self.csr.dot(self._get_value(other)))

    def __neg__(self):
        return Operator(-self.csr)

    def __mul__(self, other):
        return Operator(self.csr * self._get_value(other))

    def __rmul__(self, other):
        return Operator(self._get_value(other) * self.csr)

    def __truediv__(self, other):
        return Operator(self.csr / self._get_value(other))

    def __rtruediv__(self, other):
        return Operator(self._get_value(other) / self.csr)

    def __add__(self, other):
        return Operator(self.csr + self._get_value(other))

    def __radd__(self, other):
        return Operator(self._get_value(other) + self.csr)

    def __repr__(self):
        return f"C{self.index}{self.CHARS[self.spin]}"

    def __str__(self):
        return self.__repr__()  # str(self.dense)


def annihilation_operators(basis):
    operators = list()
    for s in range(basis.n_spins):
        ops = list()
        for i in range(basis.n_sites):
            ops.append(Operator.annihilation_operator(i, s, basis))
        operators.append(ops)
    return operators


class HamiltonOperator:
    """

    Examples
    --------
    >>> hamop = HamiltonOperator(u=u_op, eps=eps_op, t=t_op)
    >>> ham = hamop.build(u=4, eps=0, t=1)
    """

    def __init__(self, **opkwargs):
        self.operators = opkwargs

    @property
    def keys(self):
        return self.operators.keys()

    def set_operator(self, key, value):
        self.operators.update({key: value})

    def build_operator(self, key, val):
        ops = self.operators[key]
        if hasattr(ops, "__len__"):
            if not hasattr(val, "__len__"):
                val = [val] * len(ops)
            return sum([x * o for x, o in zip(val, ops)])
        else:
            return val * ops

    def build(self, **params):
        ops = list()
        for key in self.keys:
            val = params.get(key, None)
            if val is None:
                val = 0
            ops.append(self.build_operator(key, val))
        return sum(ops)

    def __str__(self):
        args = ", ".join(list(self.operators.keys()))
        return f"H({args})"
