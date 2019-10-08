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
from qsim2 import Statevector, STATES
from qsim2.gates import *


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


def initialize_qubits(arg):
    if isinstance(arg, int):
        qubits = [Qubit(0)] * arg
    else:
        qubits = [Qubit(x) for x in arg]
    return qubits


class Qubit(np.ndarray):

    def __new__(cls, arg, dtype=None):
        if isinstance(arg, int):
            arg = str(arg)
        if isinstance(arg, str):
            arg = STATES[arg]
        return np.asarray(arg, dtype).view(cls)



def main():
    qubits = initialize_qubits("100")
    q1, q2, q3 = qubits

    state = kron(qubits)
    s = Statevector(state)
    print(s)

    tensor = s.tensor()
    print(tensor)
    s.from_tensor(tensor)
    print(s)


    cnot = cnot_gate()
    res = np.einsum("i,j,k, ijkl", q1, q2, q3, cnot)
    s.from_tensor(res)
    print(s)



if __name__ == "__main__":
    main()
