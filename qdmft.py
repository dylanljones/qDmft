# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
from dmft.greens import gf_omega_fft
from dmft.twosite import impurity_gf_ref
from _qsim import kron
from _qsim.statevector import Statevector
from _qsim.gates import X_GATE, Y_GATE, Z_GATE, HADAMARD_GATE, pauli
from _qsim.gates import control_gate, rx_gate
from scitools import Matrix, Plot

eye = np.eye(2)
zeros = np.zeros((2, 2))


def xy_gate_ref(arg):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(arg), -1j*np.sin(arg), 0],
                     [0, -1j*np.sin(arg), np.cos(arg), 0],
                     [0, 0, 0, 1]])


def gf_time(tau, alpha1, alpha2, omega1, omega2):
    return 2 * (alpha1 * np.cos(omega1 * tau) + alpha2 * np.cos(omega2 * tau))


def sigma_x(*indices, n=5):
    gates = [eye] * n
    for idx in indices:
        gates[idx] = X_GATE
    return kron(gates)


def sigma_y(*indices, n=5):
    gates = [eye] * n
    for idx in indices:
        gates[idx] = Y_GATE
    return kron(gates)


def sigma_z(*indices, n=5):
    gates = [eye] * n
    for idx in indices:
        gates[idx] = Z_GATE
    return kron(gates)


def qhamiltonian(u, mu, eps_bath, v):
    h1 = u/2 * (sigma_z(1, 3) - sigma_z(1) - sigma_z(3))
    h2 = mu/2 * (sigma_z(1) + sigma_z(3))
    h3 = eps_bath/2 * (sigma_z(2) + sigma_z(4))
    h4 = v/2 * (sigma_x(1, 2) + sigma_y(1, 2) + sigma_y(3, 4) + sigma_y(3, 4))
    return h1 + h2 - h3 + h4


def circuit():
    v = 1
    tau, n = 6, 24
    taun = tau / n
    x = kron([X_GATE, np.eye(2), np.eye(2), np.eye(2)])

    s = Statevector(4)
    # Init statevector and apply x-gate on qubit 1 to set it to |1>
    s.apply_gate(x)

    # Apply xy-gate to first two qubits
    xy = np.kron(np.eye(4), xy_gate(v * taun))
    s.apply_gate(xy)

    # Apply xy-gate to last two qubits
    xy = np.kron(xy_gate(v * taun), np.eye(4))
    s.apply_gate(xy)


def xy_gate(arg):
    notc = control_gate(1, 0, X_GATE)
    crx = control_gate(0, 1, rx_gate(arg))
    xy = np.dot(notc, crx.dot(notc))
    return xy


def trotter_step(s):
    # Apply xy-gate to qubit 1 and 2
    pass


def init_xy_circuit(u, v=1):
    mu = u/2
    eps = mu
    h_gate = qu_hamiltonian(u, mu, eps, v)

    s = Statevector(5)
    s.apply_single_gate(0, HADAMARD_GATE)


def main():
    u = 3
    mu = u/2
    eps = mu
    v = 1
    s = Statevector(5)
    s.apply_single_gate(0, HADAMARD_GATE)
    s.apply_control_gate(0, 1, pauli[1])



    s.apply_control_gate(0, 1, pauli[0])
    s.apply_single_gate(0, HADAMARD_GATE)

    print(s.probabilities())
    res, p = s.measure(0)
    print(res, p)


if __name__ == "__main__":
    main()
