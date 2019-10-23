# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from .visuals import AmplitudePlot
from .utils import ZERO, Basis, to_array
from .gates import *


class Backend:

    def __init__(self, qubits, basis=None):
        self.qubits = None
        self.basis = None
        self.n_qubits = 0
        self.set_qubits(qubits, basis)

    def add_custom_gate(self, name, item):
        pass

    def set_qubits(self, qubits, basis=None):
        self.qubits = qubits
        self.n_qubits = len(qubits)
        self.basis = basis or Basis(len(qubits))

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
    GATE_DICT = GATE_DICT

    def __init__(self, qubits, basis=None, amp=None):
        super().__init__(qubits, basis)
        self.n = 2 ** self.n_qubits
        self.amp = None
        self.snapshots = list()
        self.init(amp)

    def init(self, amp=None):
        self.amp = kron([ZERO] * self.n_qubits) if amp is None else np.copy(amp)

    def set_qubits(self, qubits, basis=None):
        super().set_qubits(qubits, basis)
        self.n = 2 ** self.n_qubits
        self.init(None)

    @property
    def norm(self):
        return la.norm(self.amp)

    @property
    def last(self):
        return self.snapshots[-1]

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
        plot = AmplitudePlot(self.n_qubits)
        plot.set_amps(amps)
        if show:
            plot.show()
        return plot

    def save_snapshot(self):
        s = StateVector(self.qubits, self.basis, self.amp)
        self.snapshots.append(s)

    def add_custom_gate(self, name, item):
        self.GATE_DICT.update({name: item})

    def state(self):
        return self.amp

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

    @classmethod
    def _get_gatefunc(cls, name):
        return cls.GATE_DICT.get(name.lower())

    def build_gate(self, gate):
        if gate.is_controlled:
            name = gate.name.replace("c", "")
            gate_func = self._get_gatefunc(name)
            arr = cgate(gate.con_indices, gate.qu_indices[0], gate_func(gate.arg), self.n_qubits)
        elif gate.size > 1:
            gate_func = self._get_gatefunc(gate.name)
            arr = gate_func(gate.qu_indices, self.n_qubits, gate.arg)
        else:
            gate_func = GATE_DICT.get(gate.name.lower())
            arr = single_gate(gate.qu_indices, gate_func(gate.arg), self.n_qubits)
        return arr

    def apply_gate(self, gate, *args, **kwargs):
        if not isinstance(gate, np.ndarray):
            gate = self.build_gate(gate)
        self.amp = np.dot(gate, self.amp)

    def _measure_qubit(self, qubit):
        idx = qubit.index
        # Simulate measurement of qubit q
        projected = self.project(idx, 0)
        p0 = np.sum(np.abs(projected) ** 2)
        value = int(np.random.random() > p0)
        if value == 1:
            projected = self.project(idx, 1)
        # Project other qubits and normalize
        self.amp = projected / la.norm(projected)
        return value

    def measure(self, qbits, snapshot=True):
        if snapshot:
            self.save_snapshot()
        return [self._measure_qubit(q) for q in to_array(qbits)]
