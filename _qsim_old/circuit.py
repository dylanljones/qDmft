# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: circuit$
version: 1.0
"""

from _qsim_old import Register
from _qsim_old.objects import *


class Circuit:

    def __init__(self, regs):
        self.register = Register(regs)
        self.bits = np.zeros(self.register.n)
        self.objects = list()

    @property
    def n(self):
        return self.register.n

    def set_state(self, regs):
        self.register.state = State(regs)

    def print_header(self, header=""):
        print(header + self.register.basis_string())

    def print(self, header=""):
        print(header + self.register.register_string())

    # =========================================================================

    def add_gate(self, gate):
        self.objects.append(gate)

    def add_measurement(self, qubits=0, targets=None):
        """

        Parameters
        ----------
        qubits: array_like or int
        targets: array_like or int, optional

        Returns
        -------

        """
        if not hasattr(qubits, "__len__"):
            qubits = [qubits]
        if targets is None:
            targets = qubits
        elif not hasattr(targets, "__len__"):
            targets = [targets]
        for i in range(len(qubits)):
            qbit = qubits[i]
            bit = targets[i]
            m = Measurement(self.register, qbit, out=self.bits, target=bit)
            self.objects.append(m)

    def x(self, qubit):
        return Gate.single(self.register, qubit, X_GATE, "X")

    def y(self, qubit):
        return Gate.single(self.register, qubit, Y_GATE, "Y")

    def z(self, qubit):
        return Gate.single(self.register, qubit, Z_GATE, "Z")

    def h(self, qubit):
        return Gate.single(self.register, qubit, HADAMARD_GATE, "Hadamard")

    def cx(self, con, qubit):
        return Gate.single_control(self.register, con, qubit, X_GATE, "CX")

    def cy(self, con, qubit):
        return Gate.single_control(self.register, con, qubit, X_GATE, "CY")

    def cz(self, con, qubit):
        return Gate.single_control(self.register, con, qubit, X_GATE, "CZ")

    def rx(self, con, qubit, theta):
        gate = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        return Gate.single_control(self.register, con, qubit, gate, "RX")

    def swap(self, q1, q2):
        cxij = self.cx(q1, q2).array
        cxji = self.cx(q2, q1).array
        arr = np.dot(cxij, cxji.dot(cxij))
        return Gate(self.register, [q1, q2], arr, "Swap")

    def toffoli(self, con1, con2, qubit):
        eye = np.eye(2)
        arrs1 = [eye] * self.n
        arrs1[con1] = State.p0

        arrs2 = [eye] * self.n
        arrs2[con1] = State.p1
        arrs2[con2] = State.p0

        arrs3 = [eye] * self.register.n
        arrs3[con1] = State.p1
        arrs3[con2] = State.p1
        arrs3[qubit] = X_GATE

        arr = kron(arrs1) + kron(arrs2) + kron(arrs3)
        return Gate(self.register, qubit, arr, "Toffoli", con=[con1, con2])

    # =========================================================================

    @staticmethod
    def apply_gate(gate):
        gate.apply()

    def run(self, verbose=True):
        width = 10
        if verbose:
            self.print_header(" "*width)
            s = "Init"
            self.print(f"{s: <{width}}")
        for obj in self.objects:
            obj.apply()
            if verbose:
                s = obj.name
                self.print(f"{s: <{width}}")

    def amplitudes(self):
        arr = np.asarray(self.register.state)
        s = int(np.sqrt(arr.size))
        return arr.reshape((s, s))

    def measure_qubit(self, i, simulate=True):
        return self.register.measure_qubit(i, simulate)

    def measure(self, simulate=True):
        res = np.zeros((self.n, 2))
        for i in range(self.n):
            res[i] = self.measure_qubit(i, simulate)
        return res.T
