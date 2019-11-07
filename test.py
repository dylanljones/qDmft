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


def project(statevector, op, qubit):
    parts = [np.eye(2)] * statevector.n_qubits
    parts[qubit] = op
    proj = kron(parts)
    return proj.dot(statevector.amp)


def simulate_probability(p):
    return int(np.random.random() <= p)


def projective_measurement(s, qubit, shadow=False):
    projection = project(s, P0, qubit)
    p0 = np.dot(s.amp, projection)
    value = simulate_probability(p0)
    if value == 0:
        state = projection
    else:
        state = project(s, P1, qubit)
    if not shadow:
        s.amp = state / la.norm(state)
    return value



def general_measurement(s, qubit, op=Z_GATE, shadow=False):
    psi = s.amp

    eigvals, eigvecs = la.eig(op)
    v0, v1 = eigvecs.T

    op = np.dot(v0[:, np.newaxis], v0[np.newaxis, :])
    m = build_operator(qubit, op, s.n_qubits)
    p0 = np.sum(np.abs(m.dot(psi))**2)
    value = simulate_probability(p0)
    if value == 0:
        state = m.dot(psi)
    else:
        op = np.dot(v1[:, np.newaxis], v1[np.newaxis, :])
        m = build_operator(qubit, op, s.n_qubits)
        state = m.dot(psi)
    if not shadow:
        s.amp = state / la.norm(state)
    return eigvals[value]



def main():
    c = Circuit(2)
    c.h(0)
    # c.rx(0, arg=np.pi)
    # c.cx(0, 1)
    c.run()

    s = c.backend
    print(s)
    # print(projective_measurement(s, 0))
    values = list()
    for i in range(100):
        x = general_measurement(s, 0, shadow=True).real
        print(x)
        values.append(x)
    print("->", np.mean(values))





if __name__ == "__main__":
    main()
