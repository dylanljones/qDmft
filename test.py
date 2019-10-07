# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: gates$
version: 1.0
"""
import numpy as np
from itertools import product, permutations
from qsim import kron, Statevector
from qsim.gates import HADAMARD_GATE

ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)


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


def measure_tensor(tensor, idx):
    n = len(tensor.shape)
    indices = list(range(n))
    indices.remove(idx)
    probs = np.sum(tensor, axis=tuple(indices))
    res = np.argmax(probs)
    return res, probs[res]


def main():
    qubits = [ZERO, ZERO]
    n = len(qubits)

    qubits[0] = np.dot(HADAMARD_GATE, qubits[0])
    state = kron(qubits).squeeze()
    s = Statevector(state)
    tensor = s.tensor()
    print(s)

    print(state)
    print("-----")
    val, p = measure_tensor(tensor, 0)
    print(val, p)


    return
    cnot = cnot_gate()
    res = np.einsum("i,j,ijkl", q1, q2, cnot)

    print(res)


if __name__ == "__main__":
    main()
