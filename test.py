# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import itertools
from qsim import *
from scitools import Matrix

si, sx, sy, sz = pauli


def density_matrix(state):
    state = np.asarray(state)
    if len(state.shape) == 1:
        state = state[:, np.newaxis]
    return np.dot(state, state.T)


class Backend:

    def __init__(self, reg, basis=None):
        self.reg = None
        self.basis = None
        self.set_qubits(reg, basis)

    @property
    def n_qubits(self):
        return self.reg.n

    def add_custom_gate(self, name, item):
        pass

    def set_qubits(self, reg, basis=None):
        self.reg = reg
        self.basis = basis or Basis(reg.n)

    def state(self):
        pass

    def apply_gate(self, gate):
        pass

    def measure(self, bit):
        pass


class DensityMatrix(Backend):

    def __init__(self, reg, basis=None, state=None):
        super().__init__(reg, basis)
        self.n = 2 ** self.n_qubits
        self.rho = None
        self.init(state)

    def init(self, state=None):
        if state is None:
            state = kron([ZERO] * self.n_qubits)
        elif isinstance(state, StateVector):
            state = state.amp
        rho = density_matrix(state)
        self.rho = rho

    def __str__(self):
        string = "Density matrix:\n"
        string += str(Matrix(self.rho))
        return string

    def show(self, show=True, rotate=False):
        rot = 45 if rotate else None
        rho = Matrix(self.rho)
        return rho.show(show, labels=self.basis.labels, rotation=rot)

    def apply_gate(self, gate):
        if isinstance(gate, Gate):
            gate = gate.build_matrix(self.n_qubits)
        self.rho = np.dot(gate.T, np.dot(self.rho, gate))

    def measure(self, qubit, operator=P0):
        idx = qubit.index

        op = single_gate(idx, operator, self.n_qubits)

        res = np.trace(np.dot(self.rho, op))
        rho_new = np.dot(op, np.dot(self.rho, op))
        return res, rho_new


def build_operator(n, indices, operators):
    if isinstance(indices, int):
        indices = [indices]
        operators = [operators]
    parts = [np.eye(2)] * n
    for i, op in zip(indices, operators):
        parts[i] = op
    return kron(*parts)


def main():
    reg = QuRegister(2)
    s = StateVector(reg.bits)

    print(s)
    print(s.expectation(Z_GATE, 0))

    s.apply_gate(Gate.x(reg[0]))
    print(s.expectation(Z_GATE, 0))
    # s.apply_gate(Gate.h(reg[0]))


if __name__ == "__main__":
    main()
