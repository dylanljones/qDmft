# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from scitools import Plot
from qsim import Statevector, STATES, ONE, ZERO
from qsim.gates import *


def show_measurement_hist(tensor, idx, n=1000):
    results = np.zeros(n)
    for i in range(n):
        val, p = measure_tensor(tensor, idx)
        results[i] = val
    plot = Plot()
    plot.histogram(results)
    plot.show()


def measure_tensor(tensor, idx, decimals=10):
    n = len(tensor.shape)
    indices = list(range(n))
    indices.remove(idx)
    amps = np.sum(tensor, axis=tuple(indices))
    probs = np.abs(amps / la.norm(amps))**2
    probs = np.round(probs, decimals)
    res = np.random.choice([0, 1], p=probs)
    return res, probs[res]


def cnot_gate():
    gate = np.zeros((2, 2, 2, 2))
    gate[0, 0, 0, 0] = 1
    gate[0, 1, 0, 1] = 1
    gate[1, 0, 1, 1] = 1
    gate[1, 1, 1, 0] = 1
    return gate


def notc_gate():
    gate = np.zeros((2, 2, 2, 2))
    gate[0, 0, 0, 0] = 1
    gate[1, 0, 1, 0] = 1
    gate[0, 1, 0, 0] = 1
    gate[1, 1, 0, 1] = 1
    return gate


class Qubit:

    def __init__(self, x):
        if isinstance(x, str):
            s = STATES[x]
        else:
            s = ONE if x else ZERO
        self.state = np.asarray(s)

    def __str__(self):
        return str(self.state)

    def apply_gate(self, gate):
        self.state = np.dot(gate, self.state)

    def kron(self, other):
        return np.kron(self.state, other.state)


def main():
    q1, q2 = Qubit(1), Qubit(0)
    qubits = q1, q2
    n = len(qubits)

    # qubits[0] = np.dot(X_GATE, qubits[0])
    # qubits[0] = np.dot(HADAMARD_GATE, qubits[0])

    state = q1.kron(q2).squeeze()
    # state = kron(qubits).squeeze()
    print(state)
    s = Statevector(state)
    print(s)

    tensor = s.tensor()
    print(tensor)
    cnot = cnot_gate()
    res = np.einsum("i...,j...,ijkl", q1.state, q2.state, cnot)
    s.from_tensor(res)
    print(s)



if __name__ == "__main__":
    main()
