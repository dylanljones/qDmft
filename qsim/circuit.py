# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import re
import numpy as np
from scitools import Plot
from .utils import Basis, get_info
from .visuals import CircuitString
from .instruction import Gate, Measurement, Instruction, ParameterMap
from .backends import StateVector


def histogram(data, normalize=True):
    n, n_bins = data.shape
    binvals = np.power(2, np.arange(n_bins))[::-1]
    data = np.sum(data * binvals[np.newaxis, :], axis=1)
    hist, edges = np.histogram(data, bins=np.arange(2 ** n_bins+1))
    bins = edges[:-1].astype("int")  # + 0.5
    if normalize:
        hist = hist / n
    return bins, hist


class CircuitResult:

    def __init__(self, data, basis_labels):
        self.labels = basis_labels
        self.data = None
        self.hist = None

        self.load(data)

    def load(self, data, normalize=True):
        self.data = data
        self.hist = histogram(data, normalize)

    @property
    def shape(self):
        return self.data.shape

    @property
    def n(self):
        return self.shape[0]

    def mean(self):
        return self.sorted()[0]

    def sorted(self):
        bins, probs = self.hist
        indices = np.argsort(probs)[::-1]
        return [(bins[i], probs[i]) for i in indices]

    def highest(self, thresh=0.7):
        res_sorted = self.sorted()
        pmax = res_sorted[0][1]
        return [(self.labels[i], p) for i, p in res_sorted if p >= thresh * pmax]

    def show_histogram(self, show=True):
        bins, hist = self.hist
        plot = Plot(xlim=(-0.5, len(bins) - 0.5), ylim=(0, 1))
        plot.set_title(f"N={self.n}")
        plot.grid(axis="y")
        plot.set_ticks(bins, np.arange(0, 1.1, 0.1))
        plot.set_ticklabels(self.labels)
        plot.ax.bar(bins, hist, width=0.9)
        if show:
            plot.show()

    def __str__(self):
        entries = [f"   {label} {p:.3f}" for label, p in self.highest()]
        string = f"Result ({self.n} shots):\n"
        string += "\n".join(entries)
        return string


class Circuit:

    def __init__(self, qubits, clbits=None, backend=StateVector.name):
        self.qbits = qubits
        self.cbits = clbits or qubits
        self.basis = Basis(self.qbits)
        self.instructions = list()
        self.pmap = ParameterMap.instance()
        self.res = None

        if backend == StateVector.name:
            self.backend = StateVector(self.qbits, self.basis)
        else:
            raise ValueError("Invalid backend: " + backend)

    @classmethod
    def like(cls, other):
        return cls(other.qbits, other.cbits, other.backend.name)

    def init(self):
        self.backend.init()

    def init_params(self, *args):
        self.pmap.init(*args)

    def set_params(self, args):
        self.pmap.set_params(args)

    @property
    def num_params(self):
        return len(self.pmap.params)

    @property
    def params(self):
        return self.pmap.params

    @property
    def args(self):
        return self.pmap.args

    def append(self, circuit):
        n = len(self.instructions)
        for i, inst in enumerate(circuit.instructions):
            inst.idx = n + i
            self.instructions.append(inst)
        Instruction.INDEX = len(self.instructions)

    def add_qubit(self, idx=0):
        self.qbits += 1
        self.basis = Basis(self.qbits)
        self.backend.set_qubits(self.qbits)
        for inst in self.instructions:
            inst.insert_qubit(idx)

        if self.qbits == self.cbits + 1:
            self.add_clbit()

    def add_clbit(self):
        self.cbits += 1

    # =========================================================================

    def __repr__(self):
        return f"Circuit(qubits: {self.qbits}, clbits: {self.cbits})"

    def __str__(self):
        string = self.__repr__()
        for inst in self.instructions:
            string += "\n   " + str(inst)
        return string

    def print(self, padding=1, maxwidth=None):
        s = CircuitString(self.qbits, padding)
        for instructions in self.instructions:
            s.add(instructions)
        print(s.build(wmax=maxwidth))

    def show(self):
        pass

    def to_string(self, delim="; "):
        info = [f"qubits={self.qbits}", f"clbits={self.cbits}"]
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
            inst = Instruction.from_string(line, delim)
            self.add_instruction(inst)
        return self

    def save(self, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "w") as f:
            f.write(self.to_string(delim))

    @classmethod
    def load(cls, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "r") as f:
            string = f.read()
        return cls.from_string(string, delim)

    # =========================================================================

    def add_instruction(self, inst):
        self.instructions.append(inst)
        return inst

    def add_gate(self, name, qbits, con=None, arg=None, argidx=None):
        gates = Gate(name, qbits, con=con, arg=arg, argidx=argidx)
        return self.add_instruction(gates)

    def i(self, qubit):
        return self.add_gate("I", qubit)

    def x(self, qubit):
        return self.add_gate("X", qubit)

    def y(self, qubit):
        return self.add_gate("Y", qubit)

    def z(self, qubit):
        return self.add_gate("Z", qubit)

    def h(self, qubit):
        return self.add_gate("H", qubit)

    def s(self, qubit):
        return self.add_gate("S", qubit)

    def t(self, qubit):
        return self.add_gate("T", qubit)

    def rx(self, qubit, arg=0, argidx=None):
        return self.add_gate("Rx", qubit, arg=arg, argidx=argidx)

    def ry(self, qubit, arg=0, argidx=None):
        return self.add_gate("Ry", qubit, arg=arg, argidx=argidx)

    def rz(self, qubit, arg=0, argidx=None):
        return self.add_gate("Rz", qubit, arg=arg, argidx=argidx)

    def cx(self, con, qubit):
        return self.add_gate("X", qubit, con)

    def cy(self, con, qubit):
        return self.add_gate("Y", qubit, con)

    def cz(self, con, qubit):
        return self.add_gate("Z", qubit, con)

    def ch(self, con, qubit):
        return self.add_gate("H", qubit, con)

    def cs(self, con, qubit):
        return self.add_gate("S", qubit, con)

    def ct(self, con, qubit):
        return self.add_gate("T", qubit, con)

    def crx(self, con, qubit, arg=0, argidx=None):
        return self.add_gate("Rx", qubit, con, arg, argidx)

    def cry(self, con, qubit, arg=0, argidx=None):
        return self.add_gate("Ry", qubit, con, arg, argidx)

    def crz(self, con, qubit, arg=0, argidx=None):
        return self.add_gate("Rz", qubit, con, arg, argidx)

    def m(self, qbits=None, cbits=None):
        if qbits is None:
            qbits = range(self.qbits)
        if cbits is None:
            cbits = qbits
        self.add_instruction(Measurement("m", qbits, cbits))

    def measure(self, qbits):
        return self.backend.measure(qbits)

    def state(self):
        return self.backend.state()

    def run_shot(self, *args, **kwargs):
        self.init()
        data = np.zeros(self.cbits)
        for inst in self.instructions:
            if isinstance(inst, Gate):
                self.backend.apply_gate(inst, *args, **kwargs)
            elif isinstance(inst, Measurement):
                data = np.zeros(self.cbits)
                values = self.backend.measure(inst.qbits)
                for idx, x in zip(inst.cbits, values):
                    data[idx] = x
        return data

    def run(self, shots=1, *args, **kwargs):
        data = np.zeros((shots, self.cbits))
        for i in range(shots):
            data[i] = self.run_shot(*args, **kwargs)
        self.res = CircuitResult(data, self.basis.labels)
        return self.res

    def histogram(self):
        return self.res.hist

    def show_histogram(self, show=True):
        return self.res.show_histogram(show)
