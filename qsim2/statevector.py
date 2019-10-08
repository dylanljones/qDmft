# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from itertools import product, permutations
from scitools import Plot, Circle
from .utils import kron, basis_states, binstr, STATES, P0, P1


class AmplitudePlot(Plot):

    def __init__(self, n, lim=1.01):
        super().__init__(create=False)
        self.set_gridspec(n, n)
        self.amps = list()
        for i in range(int(n*n)):
            ax = self.add_gridsubplot(i)
            # Configure subplot
            self.set_limits((-lim, lim), (-lim, lim))
            self.set_ticklabels([], [])
            self.set_equal_aspect()

            circ = Circle((0, 0), radius=1.0, fill=False, color="k", lw=0.5)
            ax.add_artist(circ)
            self.amps.append(ax.plot([0, 1], [0, 0], lw=2)[0])
        self.set_figsize(300, ratio=1)
        self.tight()

    def set_amps(self, amps):
        for i in range(len(amps)):
            amp = amps[i]
            points = np.array([[0, 0], [amp.real, amp.imag]])
            self.amps[i].set_data(*points.T)


def initialize_state(arg):
    if isinstance(arg, int):
        state = kron([ZERO] * arg)
    elif isinstance(arg, str):
        arg = arg.strip(" ")
        states = [STATES[x] for x in arg]
        state = kron(states)
    elif isinstance(arg, list) or isinstance(arg, tuple):
        state = kron(states)
    else:
        state = arg
    return np.asarray(state, dtype="float")


class Statevector:

    def __init__(self, arg):
        self.amp = initialize_state(arg)
        self.n = self.amp.shape[0]
        self.n_qubits = int(np.log2(self.n))
        self.basis = basis_states(self.n)

    def __str__(self):
        amps = self.amplitudes(decimals=10)
        strings = [f"|{binstr(self.basis[i], self.n_qubits)}> {amps[i]}" for i in range(self.n)]
        n_max =  max([len(x) for x in strings])
        n = max(int((n_max - 6) // 2), 1)
        head = "-" * n + "Vector" + "-" * n
        strings.insert(0, head)
        # strings.append("-" * len(head))
        return "\n".join(strings) + "\n"

    def show_amplitudes(self, show=True):
        amps = self.amplitudes()
        plot = AmplitudePlot(self.n_qubits)
        plot.set_amps(amps)
        if show:
            plot.show()
        return plot

    def __getitem__(self, item):
        return self.amp[item]

    def __setitem__(self, item, value):
        self.amp[item] = value

    def normalize(self):
        self.amp /= la.norm(self.amp)

    def tensor(self, size=2):
        return self.amp.reshape([size] * self.n_qubits)

    def from_tensor(self, tensor):
        self.amp = tensor.reshape(self.n)

    def amplitudes(self, decimals=10):
        return np.round(self.amp, decimals)

    def probabilities(self, decimals=10):
        return np.abs(self.amplitudes(decimals))**2

    def project(self, idx, val):
        proj = P0 if val == 0 else P1
        parts = [np.eye(2)] * self.n_qubits
        parts[idx] = proj
        proj = kron(parts)
        return np.dot(proj, self.amp)

    def measure(self, idx, decimals=10):
        projected = self.project(idx, 0)
        prob0 = np.sum(np.round(np.abs(projected) ** 2, decimals))
        if np.random.random() <= prob0:
            res, p = 0, prob0
            self.amp = projected
        else:
            res, p = 1, 1 - prob0
            self.amp = self.project(idx, 1)
        self.normalize()
        return res, p

    def apply_gate(self, gate, idx=0):
        gate_size = gate.shape[0]
        if gate_size == 2:
            parts = [np.eye(2)] * self.n_qubits
            parts[idx] = gate
            gate = kron(parts)
        self.amp = np.dot(gate, self.amp)
