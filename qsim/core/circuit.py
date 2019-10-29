# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scitools import Plot, Terminal
from .register import Qubit, Clbit, QuRegister
from .utils import Basis, get_info, to_list, histogram
from .visuals import CircuitString
from .instruction import Gate, Measurement, Instruction, ParameterMap
from .backends import StateVector


class CircuitResult:

    def __init__(self, data):
        self.data = None
        self.basis = None
        self.hist = None
        self.load(data)

    def load(self, data, normalize=True):
        self.data = data
        self.basis = Basis(data.shape[1])
        self.hist = histogram(data, normalize)

    @property
    def shape(self):
        return self.data.shape

    @property
    def n(self):
        return self.shape[0]

    @property
    def n_bits(self):
        return self.shape[1]

    @property
    def labels(self):
        return self.basis.labels

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

    def show_histogram(self, show=True, print_values=True, max_line=True, padding=0.2,
                       color=None, alpha=0.9, lc="r", lw=1, text_padding=0, scale=False):
        bins, hist = self.hist
        plot = Plot(xlim=(-0.5, len(bins) - 0.5), ylim=(0, 1.1), title=f"N={self.n}")
        plot.grid(axis="y")
        plot.set_ticks(bins, np.arange(0, 1.1, 0.2))
        plot.set_ticklabels(self.labels)
        # plot.draw_lines(y=1, color="0.5")
        plot.bar(bins, hist, width=1-padding, color=color, alpha=0.9)

        ymax = np.max(hist)
        if print_values:
            ypos = ymax + text_padding + 0.02
            for x, y in zip(bins, hist):
                col = "0.5" if y != ymax else "0.0"
                if y:
                    plot.text((x, ypos), s=f"{y:.2f}", ha="center", va="center", color=col)
        if max_line:
            plot.draw_lines(y=ymax, color=lc, lw=lw)

        if show:
            plot.show()
        return plot

    def __str__(self):
        entries = [f"   {label} {p:.3f}" for label, p in self.highest()]
        string = f"Result ({self.n} shots):\n"
        string += "\n".join(entries)
        return string


def init_bits(arg, bit_type):
    qubits = None
    if isinstance(arg, int):
        qubits = [bit_type(i) for i in range(arg)]
    elif isinstance(arg, bit_type):
        qubits = [arg]
    elif isinstance(arg, list):
        qubits = arg
    return qubits


class Circuit:

    def __init__(self, qubits, clbits=None, backend=StateVector.name):
        self.qubits = init_bits(qubits, Qubit)
        if clbits is None:
            clbits = len(self.qubits)
        self.clbits = init_bits(clbits, Clbit)
        self.basis = Basis(self.n_qubits)
        self.instructions = list()
        self.pmap = ParameterMap.instance()
        self.res = None

        if backend == StateVector.name:
            self.backend = StateVector(self.qubits, self.basis)
        else:
            raise ValueError("Invalid backend: " + backend)

    @classmethod
    def like(cls, other):
        return cls(other.qubits, other.clbits, other.backend.name)

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def n_clbits(self):
        return len(self.clbits)

    @property
    def n_params(self):
        return len(self.pmap.params)

    @property
    def params(self):
        return self.pmap.params

    @property
    def args(self):
        return self.pmap.args

    def init(self):
        self.backend.init()

    def init_params(self, *args):
        self.pmap.init(*args)

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
        self.backend.set_qubits(self.qubits, self.basis)
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

    def _get_qubits(self, bits):
        if bits is None:
            return None
        bitlist = list()
        for q in to_list(bits):
            if not isinstance(q, Qubit):
                q = self.qubits[q]
            bitlist.append(q)
        return bitlist

    def _get_clbits(self, bits):
        if bits is None:
            return None
        bitlist = list()
        for c in to_list(bits):
            if not isinstance(c, Clbit):
                c = self.clbits[c]
            bitlist.append(c)
        return bitlist

    def add_gate(self, name, qubits, con=None, arg=None, argidx=None, n=1):
        if qubits is None:
            qubits = self.qubits
        qubits = self._get_qubits(qubits)
        con = self._get_qubits(con)
        gates = Gate(name, qubits, con=con, arg=arg, argidx=argidx, n=n)
        return self.add(gates)

    def add_measurement(self, qubits, clbits):
        qubits = self._get_qubits(qubits)
        clbits = self._get_clbits(clbits)
        m = Measurement("m", qubits, clbits)
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

    def crx(self, con, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Rx", qubit, con, arg, argidx)

    def cry(self, con, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Ry", qubit, con, arg, argidx)

    def crz(self, con, qubit, arg=np.pi/2, argidx=None):
        return self.add_gate("Rz", qubit, con, arg, argidx)

    def xy(self, qubit1, qubit2, arg=0, argidx=None):
        qubits = self._get_qubits([qubit1, qubit2])
        gate = Gate("XY", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def b(self, qubit1, qubit2, arg=0, argidx=None):
        qubits = self._get_qubits([qubit1, qubit2])
        gate = Gate("B", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def m(self, qubits=None, clbits=None):
        if qubits is None:
            qubits = range(self.n_qubits)
        if clbits is None:
            clbits = range(self.n_qubits)
        self.add_measurement(qubits, clbits)

    def measure(self, qubits):
        qubits = self._get_qubits(qubits)
        return self.backend.measure(qubits)

    def state(self):
        return self.backend.state()

    def run_shot(self, *args, **kwargs):
        self.init()
        data = np.zeros(self.n_clbits)
        for inst in self.instructions:
            if isinstance(inst, Gate):
                self.backend.apply_gate(inst, *args, **kwargs)
            elif isinstance(inst, Measurement):
                data = np.zeros(self.n_clbits)
                values = self.backend.measure(inst.qubits)
                for idx, x in zip(inst.cl_indices, values):
                    data[idx] = x
        return data

    def run(self, shots=1, verbose=False, *args, **kwargs):
        terminal = Terminal()
        header = "Running experiment"
        if verbose:
            terminal.write(header)

        data = np.zeros((shots, self.n_clbits))
        for i in range(shots):
            data[i] = self.run_shot(*args, **kwargs)
            if verbose:
                terminal.updateln(header + f": {100*(i + 1)/shots:.1f}% ({i+1}/{shots})")
        self.res = CircuitResult(data)
        if verbose:
            terminal.writeln()
            val, p = self.res.mean()
            terminal.writeln(f"Result: {val} (p={p:.2f})")
        return self.res

    def histogram(self):
        return self.res.hist

    def show_histogram(self, show=True, *args, **kwargs):
        return self.res.show_histogram(show, *args, **kwargs)
