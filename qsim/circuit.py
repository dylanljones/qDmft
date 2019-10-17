# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import re
import numpy as np
from scitools import Plot
from .utils import Grid, Basis
from .visuals import CircuitString
from .instruction import Gate, Measurement, Instruction, ParameterMap
from .backends import StateVector


def get_info(string, key, delim="; "):
    pre = key + "="
    return re.search(pre + r'(.*?)' + delim, string).group(1)


def str_to_list(string, dtype=int):
    if string.strip() == "None":
        return None
    string = string[1:-1]
    return [dtype(x) for x in string.split(" ")]


class Circuit:

    def __init__(self, qubits, clbits=None, backend=StateVector.name):
        self.qbits = qubits
        self.cbits = clbits or qubits
        self.basis = Basis(self.qbits)
        self.instructions = list()
        self.pmap = ParameterMap.instance()
        self.data = np.zeros(self.cbits)
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
        self.data = np.zeros(self.cbits)

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

    def to_qasm(self):
        n = self.qbits
        string = f"qubits {n}"
        string += f"\nprep_z q[0:{n}]"
        for inst in self.instructions:
            string += "\n  " + inst.to_qasm()
        return string

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

    def add_gate(self, name, qbits, con=None, n=1, arg=None, argidx=None):
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
        for inst in self.instructions:
            if isinstance(inst, Gate):
                self.backend.apply_gate(inst, *args, **kwargs)
            elif isinstance(inst, Measurement):
                values = self.backend.measure(inst.qbits)
                for idx, x in zip(inst.cbits, values):
                    self.data[idx] = x
        return self.data

    def run(self, shots=1, *args, **kwargs):
        data = np.zeros((shots, self.data.shape[0]))
        for i in range(shots):
            data[i] = self.run_shot(*args, **kwargs)
        self.res = data
        return self.res

    def histogram(self):
        n, nbits = self.res.shape
        binvals = np.power(2, np.arange(nbits))[::-1]
        data = np.sum(self.res * binvals[np.newaxis, :], axis=1)
        hist, edges = np.histogram(data, bins=np.arange(2 ** nbits+1))
        bins = edges[:-1] + 0.5
        return bins, hist / n

    def show_histogram(self, show=True):
        n = self.res.shape[0]
        bins, hist = self.histogram()
        idx = np.argmax(hist)
        ymax = hist[idx]
        result = self.basis.labels[idx]
        limits = np.asarray([np.max(bins) + 0.5, ymax*1.2])

        plot = Plot(ylabel="Probability", ylim=(0, limits[1]))
        plot.ax.bar(bins, hist, width=0.9)
        plot.grid(axis="y")
        plot.set_ticks(bins)
        plot.set_ticklabels(self.basis.labels)

        col = "r"
        plot.draw_lines(y=ymax, ls="-", lw=1, color=col)
        pos = (bins[idx], ymax*1.01)
        plot.ax.text(*pos, s=f"{result}\np={ymax:.2f}", color=col, ha="center", va="bottom")

        plot.text(0.99 * limits, f"N={n}", ha="right", va="top")
        if show:
            plot.show()
        return plot
