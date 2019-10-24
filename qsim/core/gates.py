# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from itertools import product
from .utils import kron, P0, P1


# ======================== SINGLE QUBIT GATES =======================

X_GATE = np.array([[0, 1], [1, 0]])
Y_GATE = np.array([[0, -1j], [1j, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
HADAMARD_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PHASE_GATE = np.array([[1, 0], [0, 1j]])
T_GATE = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])


def id_gate(args):
    return pauli[0]


def x_gate(args):
    return X_GATE


def y_gate(args):
    return Y_GATE


def z_gate(args):
    return Z_GATE


def h_gate(args):
    return HADAMARD_GATE


def s_gate(args):
    return PHASE_GATE


def t_gate(args):
    return T_GATE


def rx_gate(phi=0):
    arg = phi / 2
    return np.array([[np.cos(arg), -1j*np.sin(arg)], [-1j*np.sin(arg), np.cos(arg)]])


def ry_gate(phi=0):
    arg = phi / 2
    return np.array([[np.cos(arg), -np.sin(arg)], [+np.sin(arg), np.cos(arg)]])


def rz_gate(phi=0):
    arg = 1j * phi / 2
    return np.array([[np.exp(-arg), 0], [0, np.exp(arg)]])


# =========================================================================

def single_gate(qbits, gate, n):
    """ Builds matrix of a n-bit control gate

    Parameters
    ----------
    qbits: int or list of int
        Index of control-qubit(s)
    gate: array_like
        Matrix of the gate that is controlled
    n: int, optional
        Total number of qubits. If not specified
        number of involved qubits is used
    Returns
    -------
    gate: np.ndarray
    """
    if not hasattr(qbits, "__len__"):
        qbits = [qbits]
    eye = np.eye(2)
    arrs = list()
    for i in range(n):
        arrs.append(gate if i in qbits else eye)
    return kron(arrs)


def get_projection(*items, n):
    parts = [np.eye(2)] * n
    for idx, proj in items:
        parts[idx] = proj
    return parts


def cgate(con, t, gate, n=None):
    """ Builds matrix of a n-bit control gate

    Parameters
    ----------
    con: int or list of int
        Index of control-qubit(s)
    t: int
        Index of target qubit
    gate: array_like
        Matrix of the gate that is controlled
    n: int, optional
        Total number of qubits. If not specified
        number of involved qubits is used
    Returns
    -------
    gate: np.ndarray
    """
    if not hasattr(con, "__len__"):
        con = [con]
    n = n or max(*con, t) + 1
    gate = np.asarray(gate)
    array = 0
    for vals in product([0, 1], repeat=len(con)):
        # Projections without applying gate
        projections = [P1 if x else P0 for x in vals]
        items = list(zip(con, projections))
        if np.all(vals):
            # Projection with applying gate
            items.append((t, gate))
        # Build projection array and add to matrix
        array = array + kron(get_projection(*items, n=n))
    return array


# =========================================================================


def swap_single_cgate(gate):
    gate_tensor = gate.reshape((2, 2, 2, 2))      # reshape 4x4 gate to 2x2x2x2 tensor
    gate_tensor = np.swapaxes(gate_tensor, 0, 1)  # Switch qubit 1
    gate_tensor = np.swapaxes(gate_tensor, 2, 3)  # Switch qubit 2
    return gate_tensor.reshape((4, 4))            # reshape 2x2x2x2 tensor to 4x4 gate


def xy_gatefunc(qubits, n, arg):
    q1, q2 = qubits
    notc = cgate(q2, q1, X_GATE, n)
    crx = cgate(q1, q2, rx_gate(arg), n)
    return np.dot(notc, crx.dot(notc))


def b_gatefunc(qubits, n, arg):
    sigma = single_gate(qubits, Z_GATE, n)
    gate = np.cos(arg) * np.eye(2**n) + np.sin(arg) * sigma
    return gate


GATE_DICT = {"i": id_gate, "x": x_gate, "y": y_gate, "z": z_gate,
             "h": h_gate, "s": s_gate, "t": t_gate,
             "rx": rx_gate, "ry": ry_gate, "rz": rz_gate,
             "xy": xy_gatefunc, "b": b_gatefunc
             }
