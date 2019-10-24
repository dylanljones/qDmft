# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from scipy import sparse
from scitools import Plot
from qsim import QuRegister, StateVector


def measure_tensor(tensor, idx, decimals=10):
    n = len(tensor.shape)
    indices = list(range(n))
    indices.remove(idx)
    amps = np.sum(tensor, axis=tuple(indices))
    probs = np.abs(amps / la.norm(amps))**2
    probs = np.round(probs, decimals)
    res = np.random.choice([0, 1], p=probs)
    return res, probs[res]


def old():
    s = StateVector(2)
    print(s)

    tensor = s.tensor()
    print(tensor)
    s.from_tensor(tensor)
    print(s)

    cnot = cnot_gate()
    # res = np.einsum("i,j,k, ijkl", q1, q2, q3, cnot)
    # s.from_tensor(res)
    print(s)


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


class SparseArray:

    def __init__(self, shape):
        self.shape = shape
        self.elements = dict()

    def add(self, item, value):
        self.elements.update({item: value})

    def todense(self):
        array = np.zeros(self.shape)
        for idx, value in self.elements.items():
            array[idx] = value
        return array


def main():
    reg = QuRegister(2)
    # t = cnot_gate()
    indices = np.array([[0, 0, 0, 0],
                        [0, 1, 0, 1],
                        [1, 0, 1, 1],
                        [1, 1, 1, 0]])
    data = np.ones(4)

    t = sparse.coo_matrix((data, *indices.T), shape=(2, 2, 2, 2))
    print(t.toarray())


    # s = StateVector(reg.bits)
    # print(t.reshape((4, 4)))






if __name__ == "__main__":
    main()
