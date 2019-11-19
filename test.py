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
from scipy import sparse
from scitools import prange
from qsim.core import *
from qsim import Circuit, Gate, prepare_ground_state, test_vqe
from qsim.fermions import FBasis, Operator, HamiltonOperator

si, sx, sy, sz = pauli


def norm(a):
    return a / la.norm(a)


class SubCircuit:

    def __init__(self, name):
        self.name = name
        self.gates = list()

    @property
    def qubits(self):
        qubits = list()
        for gate in self.gates:
            print(gate.qubits)
        return qubits

    def add_gate(self, gate):
        self.gates.append(gate)


def main():
    c = Circuit(4)
    reg = c.qureg

    vqe = SubCircuit("VQE")
    vqe.add_gate(Gate.ry(reg[0, 1, 2, 3]))
    vqe.add_gate(Gate.x(reg[3], con=reg[2]))
    vqe.add_gate(Gate.ry(reg[2, 3]))
    vqe.add_gate(Gate.x(reg[2], con=reg[0]))
    vqe.add_gate(Gate.ry(reg[0, 2]))
    vqe.add_gate(Gate.x(reg[1], con=reg[0]))
    print(vqe.qubits)

    s = 0.5 * (kron(sx, sx) + kron(sy, sy)).real
    print(s)
    print(s**2)


if __name__ == "__main__":
    main()
