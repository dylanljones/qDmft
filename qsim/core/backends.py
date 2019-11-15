# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from .utils import ZERO, ONE, Basis, to_array, kron
from .register import Qubit, QuRegister
from .gates import GATE_DICT


class Backend:

    def __init__(self, qubits, basis=None):
        self.qubits = None
        self.basis = None
        self.n_qubits = 0
        self.set_qubits(qubits, basis)

    def add_custom_gate(self, name, item):
        pass

    def set_qubits(self, qubits, basis=None):
        if isinstance(qubits, QuRegister):
            qubits = qubits.bits
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
        n = max(int((n_max - 6) // 2), 1) + 1
        head = "-" * n + "Vector" + "-" * n
        strings.insert(0, head)
        return "\n".join(strings) + "\n"

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

    def density_matrix(self):
        return np.dot(self.amp[:, np.newaxis], self.amp[np.newaxis, :])

    def amplitudes(self, decimals=10):
        return np.round(self.amp, decimals)

    def probabilities(self, decimals=10):
        return np.abs(self.amplitudes(decimals))**2

    def project(self, idx, op):
        parts = [np.eye(2)] * self.n_qubits
        parts[idx] = op
        return np.dot(kron(parts), self.amp)

    def apply_unitary(self, u):
        self.amp = np.dot(u, self.amp)

    def apply_gate(self, gate, *args, **kwargs):
        if not isinstance(gate, np.ndarray):
            gate = gate.build_matrix(self.n_qubits)  # self.build_gate(gate)
        self.apply_unitary(gate)

    def measure_qubit(self, qubit, basis=None, shadow=False):
        r""" Measure the state of a single qubit in a given eigenbasis.

        The probability .math:'p_i' of measuring each eigenstate of the measurement-eigenbasis
        is calculated using the projection .math:'P_i' of the corresponding eigenvector .math:'v_i'

        .. math::
            p_i = <\Psi| P_i | \Psi > \quad P_i = |v_i > < v_i|

        The calculated probabilities are used to determine the corresponding eigenvalue .math:'\lambda_i'
        which is the final measurement result. The state after the measurement is defined as:

        .. math::
            | \Psi_{\text{new}} > = \frac{P_i | \Psi >}{\norm{P_i | \Psi >}}

        Parameters
        ----------
        qubit: Qubit
            The qubit that is measured
        basis: (2, 2) ndarray, optional
            The basis in which is measured. The default is the computational basis with
            eigenvalues '0' and '1'
        shadow: bool, optional
            Flag if state should remain in the pre-measurement state.
            The default is 'False'.

        Returns
        -------
        result: complex or float
            Eigenvalue corresponding to the measured eigenstate.
        """
        idx = qubit.index
        # get eigenbasis of measurment operator.
        # If not specified the computational basis is used
        if basis is None:
            v0, v1 = ZERO, ONE
            eigvals = [0, 1]
        else:
            eigvals, eigvecs = la.eig(basis)
            v0, v1 = eigvecs.T
        # Calculate probability of getting first eigenstate as result
        op0 = np.dot(v0[:, np.newaxis], v0[np.newaxis, :])
        projected = self.project(idx, op0)
        p0 = np.dot(self.amp, projected)
        # Simulate measurement probability
        index = int(np.random.random() > p0)
        if index == 1:
            # Project state to other eigenstate of the measurement basis
            op1 = np.dot(v1[:, np.newaxis], v1[np.newaxis, :])
            projected = self.project(idx, op1)
        # Project measurement result on the state
        if not shadow:
            self.amp = projected / la.norm(projected)
        # return corresponding eigenvalue of the measured eigenstate
        return eigvals[index]

    def measure(self, qbits, basis=None, snapshot=True, shadow=False):
        if snapshot:
            self.save_snapshot()
        return [self.measure_qubit(q, basis, shadow) for q in to_array(qbits)]

    def histogram(self):
        return np.arange(self.n), np.abs(self.amp)
