# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from qsim import *


def ham_kin_even(n):
    return single_gate(n-1, X_GATE, n)


def disorder_time_evolution(c, dt=1):
    c.rx(-1, arg=2*dt)
    for i in range(c.n_qubits-1):
        c.cx(range(i, c.n_qubits), i)
    c.x(-1)
    c.rx(-1, arg=2*dt)
    c.x(-1)
    for i in reversed(range(c.n_qubits-1)):
        c.cx(range(i, c.n_qubits), i)


def hamiltonian(n_qubits):
    n = n_qubits**2
    ham = np.zeros((n, n))
    for i in range(n-1):
        ham[i, i+1] = ham[i+1, i] = -1
    return ham

def config_vqe_circuit(c):
    c.ry([0, 1, 2])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


def prepare_groundstate(n_qubits):
    print("Preparing ground-state")
    ham = hamiltonian(3)
    vqe = VqeSolver(ham)
    config_vqe_circuit(vqe.circuit)
    sol = vqe.minimize()
    return vqe.circuit.state.amp, sol



def main():
    n_qubits = 3
    sites = 2**n_qubits
    print(f"Sites: {sites}")

    prepare_groundstate(n_qubits)

    c = Circuit(n_qubits)



    disorder_time_evolution(c)

    print(c)
    print(ham_kin_even(n_qubits))







if __name__ == "__main__":
    main()
