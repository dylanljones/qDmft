# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: statevector$
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scitools import Plot, Circle
from .states import ZERO, ONE, PLUS, MINUS, P0, P1
from .utils import kron
from .gates import single_gate, control_gate


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


def basis_states(n):
    return list(range(int(n)))


def binstr(x, n):
    return f"{bin(x)[2:]:0>{n}}"


def _init_state(x):
    if isinstance(x, int):
        initial_state = kron([ZERO] * x)
    elif isinstance(x, str):
        x = x.strip(" ")
        states = list()
        for bit in x:
            if bit == "0":
                states.append(ZERO)
            elif bit == "1":
                states.append(ONE)
            elif bit == "+":
                states.append(PLUS)
            elif bit == "-":
                states.append(MINUS)
        initial_state = kron(states)
    elif isinstance(x, list):
        initial_state = kron(x).squeeze()
    else:
        initial_state = np.asarray(x)
        if len(initial_state.shape) == 1:
            initial_state = np.asarray([x]).T
    return np.asarray(initial_state, dtype="complex")


class Statevector:

    def __init__(self, x):
        self.amp = _init_state(x)
        self.n = self.amp.shape[0]
        self.n_qubits = int(np.log2(self.n))
        self.basis = basis_states(self.n)

    def __str__(self):
        line = "-------"
        strings = [line]
        for i in range(len(self.basis)):
            strings.append(f"|{binstr(self.basis[i], self.n_qubits)}> {self.amp[i]}")
        strings.append(line)
        return "\n".join(strings)

    def __getitem__(self, item):
        return self.amp[item, 0]

    def __setitem__(self, item, value):
        self.amp[item, 0] = value

    def amplitudes(self, decimals=10):
        arr = np.asarray(self.amp).T[0]
        return np.round(arr, decimals)

    def tensor(self, size=2):
        return self.amp.reshape([size] * self.n_qubits)

    @property
    def norm(self):
        return la.norm(self.amp)

    def normalize(self):
        self.amp /= self.norm

    def show_amplitudes(self, show=True):
        amps = self.amplitudes()
        plot = AmplitudePlot(self.n_qubits)
        plot.set_amps(amps)
        if show:
            plot.show()
        return plot

    def apply_gate(self, x):
        self.amp = np.dot(x, self.amp)

    def apply_single_gate(self, idx, gate):
        x = single_gate(idx, gate, n=self.n_qubits)
        self.apply_gate(x)

    def apply_control_gate(self, c, t, gate):
        x = control_gate(c, t, gate, n=self.n_qubits)
        self.apply_gate(x)

    def probabilities(self, decimals=10):
        return np.abs(self.amplitudes(decimals))**2

    def project(self, idx, projection):
        parts = [np.eye(2)] * self.n_qubits
        parts[idx] = projection
        projection = kron(parts)
        return np.dot(projection, self.amp)

    def measure(self, idx, decimals=10):
        projected = self.project(idx, P0)
        prob0 = np.sum(np.round(np.abs(projected) ** 2, decimals))
        if np.random.random() <= prob0:
            res, p = 0, prob0
            self.amp = projected
        else:
            res, p = 1, 1 - prob0
            self.amp = self.project(idx, P1)
        self.normalize()
        return res, p
