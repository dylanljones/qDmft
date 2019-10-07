# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from qsim.gates import *

ZERO = np.array([1, 0])
ONE = np.array([0, 1])


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
