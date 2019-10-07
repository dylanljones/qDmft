# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: gates$
version: 1.0
"""
import numpy as np
from .utils import kron
from .states import P0, P1

eye2 = np.eye(2)
zero2 = np.zeros((2, 2))

# ======================== SINGLE QUBIT GATES =======================

X_GATE = np.array([[0, 1], [1, 0]])
Y_GATE = np.array([[0, -1j], [1j, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
HADAMARD_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PHASE_GATE = np.array([[1, 0], [0, 1j]])
T_GATE = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])

pauli = X_GATE, Y_GATE, Z_GATE


def eyes(n, size=2):
    return [np.eye(size)] * n


def single_gate(idx, gate, n):
    arrs = eyes(n)
    arrs[idx] = gate
    return kron(arrs)


# ---------------- ROTATION GATES -----------------------------------


def rx_gate(phi=0):
    arg = phi / 2
    return np.array([[np.cos(arg), -1j*np.sin(arg)],
                     [-1j*np.sin(arg), np.cos(arg)]])


def ry_gate(phi=0):
    arg = phi / 2
    return np.array([[np.cos(arg), -np.sin(arg)],
                     [+np.sin(arg), np.cos(arg)]])


def rz_gate(phi=0):
    arg = 1j * phi / 2
    return np.array([[np.exp(-arg), 0], [0, np.exp(arg)]])


# ======================== TWO QUBIT GATES ===========================

def get_projection(*items, n):
    parts = eyes(n)
    for idx, proj in items:
        parts[idx] = proj
    return parts


def control_gate(control, target, gate, n=2):
    parts = get_projection((control, P0), n=n)
    parts2 = get_projection((control, P1), (target, gate), n=n)
    return kron(parts) + kron(parts2)


def swap_single_cgate(gate):
    gate_tensor = gate.reshape((2, 2, 2, 2))      # reshape 4x4 gate to 2x2x2x2 tensor
    gate_tensor = np.swapaxes(gate_tensor, 0, 1)  # Switch qubit 1
    gate_tensor = np.swapaxes(gate_tensor, 2, 3)  # Switch qubit 2
    return gate_tensor.reshape((4, 4))            # reshape 2x2x2x2 tensor to 4x4 gate


def cc_gate(c1, c2, t, gate, n):
    arrs1 = get_projection((c1, P0), (c2, P0), n=n)
    arrs2 = get_projection((c1, P0), (c2, P1), n=n)
    arrs3 = get_projection((c2, P0), (c1, P1), n=n)
    arrs4 = get_projection((c1, P1), (c2, P1), (t, gate), n=n)
    return kron(arrs1) + kron(arrs2) + kron(arrs3) + kron(arrs4)
