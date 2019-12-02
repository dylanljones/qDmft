# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scitools import Terminal
from .register import Qubit, Clbit, QuRegister, ClRegister
from .utils import Basis, get_info, Result
from .backends import StateVector
from .visuals import CircuitString
from .instruction import Instruction, ParameterMap, Gate, Measurement


# def init_bits(arg, bit_type):
#     qubits = None
#     if isinstance(arg, int):
#         qubits = [bit_type(i) for i in range(arg)]
#     elif isinstance(arg, bit_type):
#         qubits = [arg]
#     elif isinstance(arg, list):
#         qubits = arg
#     return qubits


class Circuit:

    def __init__(self, qubits, clbits=None):
        self.qureg = QuRegister(qubits)
        if clbits is None:
            clbits = len(self.qubits)
        self.clreg = ClRegister(clbits)
        self.basis = Basis(self.n_qubits)
        self.instructions = list()
        self.pmap = ParameterMap.instance()
        self.res = None

        self.state = StateVector(self.qubits, self.basis)

    @classmethod
    def like(cls, other):
        return cls(other.qubits, other.clbits)

    @property
    def qubits(self):
        return self.qureg.bits

    @property
    def n_qubits(self):
        return self.qureg.n

    @property
    def clbits(self):
        return self.clreg.bits

    @property
    def n_clbits(self):
        return self.clreg.n

    @property
    def n_params(self):
        return len(self.pmap.params)

    @property
    def params(self):
        return self.pmap.params

    @property
    def args(self):
        return self.pmap.args

    def set_state(self, psi):
        self.state.set(psi)

    def init_params(self, *args):
        self.pmap.init(*args)

    def init(self, state):
        self.state.set(state)

    def set_params(self, args):
        self.pmap.set(args)

    def set_param(self, idx, arg):
        self.pmap[idx] = arg

    def __getitem__(self, item):
        return self.instructions[item]

    def __iter__(self):
        for inst in self.instructions:
            yield inst

    def append(self, circuit):
        for inst in circuit:
            self.add(inst)

    # =========================================================================

    def to_string(self, delim="; "):
        info = [f"qubits={self.n_qubits}", f"clbits={self.n_clbits}"]
        string = "".join([x + delim for x in info])
        lines = [string]
        for inst in self.instructions:
            string = inst.to_string()
            lines.append(string)
        return "\n".join(lines)

    @classmethod
    def from_string(cls, string, delim="; "):
        lines = string.splitlines()
        info = lines.pop(0)
        qbits = int(get_info(info, "qubits", delim))
        cbits = int(get_info(info, "clbits", delim))
        self = cls(qbits, cbits)
        for line in lines:
            inst = Instruction.from_string(line, self.qubits, self.clbits, delim)
            self.add(inst)
        return self

    def save(self, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "w") as f:
            f.write(self.to_string(delim))
        return file

    @classmethod
    def load(cls, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "r") as f:
            string = f.read()
        return cls.from_string(string, delim)

    def add_qubit(self, idx=None, add_clbit=False):
        if idx is None:
            idx = self.n_qubits
        new = Qubit(idx)
        for q in self.qubits:
            if q.index >= idx:
                q.index += 1
        self.qubits.insert(idx, new)
        self.basis = Basis(self.n_qubits)
        self.state.set_qubits(self.qubits, self.basis)
        if add_clbit:
            self.add_clbit(idx)

    def add_clbit(self, idx=None):
        if idx is None:
            idx = self.n_qubits
        new = Clbit(idx)
        for c in self.clbits:
            if c.index >= idx:
                c.index += 1
        self.clbits.insert(idx, new)

    def add_custom_gate(self, name, item):
        Gate.add_custom_gate(name, item)

    # =========================================================================

    def __repr__(self):
        return f"Circuit(qubits: {self.qubits}, clbits: {self.clbits})"

    def __str__(self):
        string = self.__repr__()
        for inst in self.instructions:
            string += "\n   " + str(inst)
        return string

    def print(self, show_args=True, padding=1, maxwidth=None):
        s = CircuitString(len(self.qubits), padding)
        for instructions in self.instructions:
            s.add(instructions, show_arg=show_args)
        print(s.build(wmax=maxwidth))

    def show(self):
        pass

    # =========================================================================

    def add(self, inst):
        self.instructions.append(inst)
        return inst

    def add_gate(self, name, qubits, con=None, arg=None, argidx=None, n=1, trigger=1):
        if qubits is None:
            qubits = self.qubits
        qubits = self.qureg.list(qubits)
        con = self.qureg.list(con)
        gates = Gate(name, qubits, con, arg, argidx, n, trigger)
        return self.add(gates)

    def add_measurement(self, qubits, clbits, basis=None):
        if qubits is None:
            qubits = range(self.n_qubits)
        if clbits is None:
            clbits = qubits
        qubits = self.qureg.list(qubits)
        clbits = self.clreg.list(clbits)
        m = Measurement("m", qubits, clbits, basis=basis)
        return self.add(m)

    def i(self, qubit=None):
        return self.add_gate("I", qubit)

    def x(self, qubit=None):
        return self.add_gate("X", qubit)

    def y(self, qubit=None):
        return self.add_gate("Y", qubit)

    def z(self, qubit=None):
        return self.add_gate("Z", qubit)

    def h(self, qubit=None):
        return self.add_gate("H", qubit)

    def s(self, qubit=None):
        return self.add_gate("S", qubit)

    def t(self, qubit=None):
        return self.add_gate("T", qubit)

    def rx(self, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Rx", qubit, arg=arg, argidx=argidx)

    def ry(self, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Ry", qubit, arg=arg, argidx=argidx)

    def rz(self, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Rz", qubit, arg=arg, argidx=argidx)

    def cx(self, con, qubit, trigger=1):
        return self.add_gate("X", qubit, con, trigger=trigger)

    def cy(self, con, qubit, trigger=1):
        return self.add_gate("Y", qubit, con, trigger=trigger)

    def cz(self, con, qubit, trigger=1):
        return self.add_gate("Z", qubit, con, trigger=trigger)

    def ch(self, con, qubit, trigger=1):
        return self.add_gate("H", qubit, con, trigger=trigger)

    def cs(self, con, qubit, trigger=1):
        return self.add_gate("S", qubit, con, trigger=trigger)

    def ct(self, con, qubit, trigger=1):
        return self.add_gate("T", qubit, con, trigger=trigger)

    def crx(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        return self.add_gate("Rx", qubit, con, arg, argidx, trigger=trigger)

    def cry(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        return self.add_gate("Ry", qubit, con, arg, argidx, trigger=trigger)

    def crz(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        return self.add_gate("Rz", qubit, con, arg, argidx, trigger=trigger)

    def xy(self, qubits, arg=0, argidx=None):
        if not hasattr(qubits[0], "__len__"):
            qubits = [qubits]
        qubits = [self.qureg.list(pair) for pair in qubits]
        gate = Gate("XY", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def b(self, qubits, arg=0, argidx=None):
        if not hasattr(qubits[0], "__len__"):
            qubits = [qubits]
        qubits = [self.qureg.list(pair) for pair in qubits]
        gate = Gate("B", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def m(self, qubits=None, clbits=None):
        self.add_measurement(qubits, clbits)

    def mx(self, qubits=None, clbits=None):
        self.add_measurement(qubits, clbits, "x")

    def my(self, qubits=None, clbits=None):
        self.add_measurement(qubits, clbits, "y")

    def mz(self, qubits=None, clbits=None):
        self.add_measurement(qubits, clbits, "z")

    # =========================================================================

    def expectation(self, operator):
        return self.state.expectation(operator)

    def measure(self, qubits, basis=None):
        qubits = self.qureg.list(qubits)
        return self.state.measure(qubits, basis)

    def apply_gate(self, inst):
        self.state.apply_gate(inst)

    def run_shot(self):
        data = np.zeros(self.n_clbits, dtype="float")
        for inst in self.instructions:
            if isinstance(inst, Gate):
                self.state.apply_gate(inst)
            elif isinstance(inst, Measurement):
                eigvals, eigvecs = inst.eigenbasis()
                values = self.state.measure(inst.qubits, eigvals, eigvecs)
                for idx, x in zip(inst.cl_indices, values):
                    data[idx] = x
        return data

    def run(self, shots=1, verbose=False, state0=None, *args, **kwargs):
        terminal = Terminal()
        header = "Running experiment"
        if verbose:
            terminal.write(header)
        data = np.zeros((shots, self.n_clbits), dtype="float")
        for i in range(shots):
            self.init(state0)
            data[i] = self.run_shot()
            if verbose:
                terminal.updateln(header + f": {100*(i + 1)/shots:.1f}% ({i+1}/{shots})")
        if verbose:
            terminal.writeln()
            terminal.writeln(f"Result: {np.mean(data, axis=0)}")
        return data


    def histogram(self):
        return self.res.hist

    def show_histogram(self, show=True, *args, **kwargs):
        return self.res.show_histogram(show, *args, **kwargs)
