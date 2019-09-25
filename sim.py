# -*- coding: utf-8 -*-
"""
Created on 22 Sep 2019
author: Dylan Jones

project: Qsim
version: 1.0
"""
import numpy as np
import scipy.linalg as la
import itertools
from qsim import Gate, Register, CircuitObject
from qsim.objects import *


class Measurement(CircuitObject):

    def __init__(self, register, qubit, out, target=None, decimals=5):
        if target is None:
            target = qubit
        super().__init__(register, qubit, "Measure", out)
        self.decimals = decimals
        self.idx = target

    def apply(self, simulate=True):
        res, p = self.reg.measure_qubit(self.qubit, simulate=simulate, decimals=self.decimals)
        self.out[self.idx] = res


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
        return Gate.single_control(self.register, con, qubit, X_GATE, "CNOT")

    def cy(self, con, qubit):
        return Gate.single_control(self.register, con, qubit, X_GATE, "CNOT")

    def cz(self, con, qubit):
        return Gate.single_control(self.register, con, qubit, X_GATE, "CNOT")

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

    def measure_qubit(self, i, simulate=True):
        return self.register.measure_qubit(i, simulate)

    def measure(self, simulate=True):
        res = np.zeros((self.n, 2))
        for i in range(self.n):
            res[i] = self.measure_qubit(i, simulate)
        return res.T


def truth_table(c):
    print("Input  Output")
    inputs = list(itertools.product(["0", "1"], repeat=c.n))
    for inp in inputs:
        c.set_state("".join(inp))
        c.run(verbose=False)
        res, p = c.measure()
        res = ", ".join([str(int(x)) for x in res])
        print(", ".join(inp), " ", res)


def teleportation_circuit():
    c = Circuit("00")
    c.add_gate(c.cx(0, 1))
    c.add_gate(c.h(0))
    c.add_measurement([0, 1])
    c.add_gate(c.cx(1, 2))
    c.add_gate(c.cz(0, 2))

    c.run(False)
    c.add_measurement([0, 1])
    print(c.register)
    print(c.measure())


def rz_gate(phi=np.pi):
    arg = 1j * phi / 2
    return np.array([[np.exp(-arg), 0], [0, np.exp(arg)]])


def main():
    circ = Circuit("00")

    circ.add_gate(circ.h(0))
    circ.run()
    circ.measure(1)
    circ.print()



    return
    circ = Circuit("0000")

    phi = 0.01 * np.pi

    circ.add_gate(circ.h(0))
    circ.add_gate(circ.h(1))
    circ.add_gate(circ.cx(0, 3))
    circ.add_gate(circ.cx(0, 2))

    circ.add_gate(Gate.single(circ.register, 3, rz_gate(2*phi)))

    circ.add_gate(circ.cx(0, 3))
    circ.add_gate(circ.cx(0, 2))
    circ.add_gate(circ.h(1))
    circ.add_gate(circ.h(0))

    for _ in range(20):
        circ.run(False)

    res, prob = circ.register.measure()
    print(res, prob)
    # teleportation_circuit()


if __name__ == "__main__":
    main()
