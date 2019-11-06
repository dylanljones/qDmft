# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import scipy.linalg as la
import itertools
from scitools import Matrix
from qsim import *

si, sx, sy, sz = pauli


def build_operator(indices, operators, n):
    parts = [np.eye(2)] * n
    if isinstance(indices, int):
        indices = [indices]
        operators = [operators]
    for idx, op in zip(indices, operators):
        parts[idx] = op
    return kron(parts)


def measure_comp_basis(s, qubit):
    state = s.amp
    n = s.n_qubits

    op = np.eye(2)
    eigvals, eigvecs = la.eig(op)
    v0, v1 = eigvecs.T

    # project state to result 0
    op0 = np.dot(v0[:, np.newaxis], v0[np.newaxis, :])

    proj_op = build_operator(qubit, op0, n)
    projected = proj_op.dot(state)
    prob = np.sum(np.abs(projected)**2)
    print(prob)
    print(np.abs(np.dot(state, projected)))
    print(expectation(build_operator(0, Z_GATE, n), state))


def main():
    c = Circuit(2)
    c.h(0)
    # c.rx(0, arg=np.pi)
    # c.cx(0, 1)
    c.run()


    s = c.backend
    print(s)
    measure_comp_basis(s, 0)






if __name__ == "__main__":
    main()
