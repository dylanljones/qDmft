# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from .visuals import AmplitudePlot
from .utils import ZERO, Basis, to_array
from .gates import *


class Backend:

    def __init__(self, qubits, basis=None):
        self.nbits = qubits
        self.basis = basis or Basis(qubits)

    def state(self):
        pass

    def apply_gate(self, gate):
        pass

    def measure(self, bit):
        pass


# =========================================================================
#                             STATEVECTOR
# =========================================================================


class StateVector(Backend):

    name = "statevector"

    def __init__(self, qubits, basis=None):

        super().__init__(qubits, basis)
        self.n = 2 ** qubits
        self.amp = None
        self.init()

    def init(self):
        self.amp = kron([ZERO] * self.nbits)

    @property
    def norm(self):
        return la.norm(self.amp)

    def __getitem__(self, item):
        return self.amp[item]

    def __setitem__(self, item, value):
        self.amp[item] = value

    def __str__(self):
        amps = self.amplitudes(decimals=10)
        strings = [f"{self.basis.labels[i]} {amps[i]}" for i in range(self.n)]
        n_max = max([len(x) for x in strings])
        n = max(int((n_max - 6) // 2), 1)
        head = "-" * n + "Vector" + "-" * n
        strings.insert(0, head)
        return "\n".join(strings) + "\n"

    def show_amplitudes(self, show=True):
        amps = self.amplitudes()
        plot = AmplitudePlot(self.nbits)
        plot.set_amps(amps)
        if show:
            plot.show()
        return plot

    def state(self):
        return self.amp

    def normalize(self):
        self.amp /= la.norm(self.amp)

    def tensor(self, size=2):
        return self.amp.reshape([size] * self.nbits)

    def from_tensor(self, tensor):
        self.amp = tensor.reshape(self.n)

    def amplitudes(self, decimals=10):
        return np.round(self.amp, decimals)

    def probabilities(self, decimals=10):
        return np.abs(self.amplitudes(decimals))**2

    def project(self, idx, val):
        proj = P0 if val == 0 else P1
        parts = [np.eye(2)] * self.nbits
        parts[idx] = proj
        proj = kron(parts)
        return np.dot(proj, self.amp)

    def build_gate(self, gate):
        arr = np.eye(self.n)
        if gate.is_controlled:
            name = gate.name.replace("c", "")
            gate_func = GATE_DICT.get(name.lower())
            arr = cgate(gate.con, gate.qbits[0], gate_func(gate.arg), self.nbits)
        else:
            gate_func = GATE_DICT.get(gate.name.lower())
            arr = single_gate(gate.qbits, gate_func(gate.arg), self.nbits)
        return arr

    def apply_gate(self, gate, *args, **kwargs):
        arr = self.build_gate(gate)
        self.amp = np.dot(arr, self.amp)

    def _measure_qubit(self, qbit):
        # Simulate measurement of qubit q
        projected = self.project(qbit, 0)
        p0 = np.sum(np.abs(projected) ** 2)
        value = int(np.random.random() > p0)
        if value == 1:
            projected = self.project(qbit, 1)
        # Project other qubits and normalize
        self.amp = projected / la.norm(projected)
        return value

    def measure(self, qbits):
        return [self._measure_qubit(q) for q in to_array(qbits)]
