# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import re
import numpy as np
from .params import ParameterMap
from .utils import to_list


def str_to_list(s, dtype=int):
    if s.strip() == "None":
        return None
    return [dtype(x) for x in re.findall(r'(\d+(?:\.\d+)?)', s)]


class Instruction:

    INDEX = 0
    TYPE = "Instruction"
    pmap = ParameterMap.instance()

    def __init__(self, name, qubits=None, con=None, clbits=None, n=1, arg=None, argidx=None):
        self.idx = Instruction.INDEX
        Instruction.INDEX += 1
        self.size = 1
        self.name = name

        self.qubits = to_list(qubits) if qubits is not None else None
        self.con = to_list(con) if con is not None else None
        self.clbits = to_list(clbits) if clbits is not None else None

        self.pmap.add(arg, argidx)

    @property
    def is_controlled(self):
        return bool(self.con)

    @property
    def n_qubits(self):
        return len(self.qubits) if self.qubits is not None else 0

    @property
    def qu_indices(self):
        return [q.index for q in self.qubits] if self.qubits is not None else None

    @property
    def n_con(self):
        return len(self.con) if self.con is not None else 0

    @property
    def con_indices(self):
        return [q.index for q in self.con] if self.con is not None else None

    @property
    def n_clbits(self):
        return len(self.clbits) if self.clbits is not None else 0

    @property
    def cl_indices(self):
        return [c.index for c in self.clbits] if self.clbits is not None else None

    @property
    def argidx(self):
        return self.pmap.indices[self.idx] #.index(self.idx)

    @property
    def arg(self):
        return self.pmap.get(self.idx)

    def _attr_str(self):
        parts = [self.name, f"ID: {self.idx}"]
        if self.n_qubits:
            parts.append(f"qBits: {self.n_qubits}")
        if self.n_con:
            parts.append(f"con: {self.n_con}")
        if self.n_clbits:
            parts.append(f"cBits: {self.n_clbits}")
        if self.arg is not None:
            parts.append(f"Args: {self.arg}")
        return ", ".join(parts)

    def __str__(self):
        return f"{self.TYPE}({self._attr_str()})"

    def set_arg(self, value):
        self.pmap.set(self.idx, value)

    def to_dict(self):
        return dict(idx=self.idx, name=self.name, qbits=self.qu_indices,
                    con=self.con_indices, cbits=self.cl_indices, arg=self.arg,
                    argidx=self.argidx)

    def to_string(self, delim="; "):
        string = ""
        for key, val in self.to_dict().items():
            string += f"{key}={val}{delim}"
        return string


class Measurement(Instruction):

    TYPE = "Measurement"

    def __init__(self, name, qubits, clbits=None):
        if clbits is None:
            clbits = qubits
        super().__init__(name, qubits, clbits=clbits)


class Gate(Instruction):

    TYPE = "Gate"

    def __init__(self, name, qubits, con=None, arg=None, argidx=None):
        super().__init__(name, qubits, con=con, arg=arg, argidx=argidx)
        if con is not None:
            self.name = "c" * len(self.con) + self.name

    @classmethod
    def x(cls, qubits, con=None):
        return cls("X", qubits, con)

    @classmethod
    def y(cls, qubits, con=None):
        return cls("Y", qubits, con)

    @classmethod
    def z(cls, qubits, con=None):
        return cls("Z", qubits, con)

    @classmethod
    def h(cls, qubits, con=None):
        return cls("H", qubits, con)

    @classmethod
    def s(cls, qubits, con=None):
        return cls("S", qubits, con)

    @classmethod
    def t(cls, qubits, con=None):
        return cls("T", qubits, con)

    @classmethod
    def rx(cls, qubit, arg=0, argidx=None, con=None):
        return cls("Rx", qubit, con, arg, argidx)

    @classmethod
    def ry(cls, qubit, arg=0, argidx=None, con=None):
        return cls("Ry", qubit, con, arg, argidx)

    @classmethod
    def rz(cls, qubit, arg=0, argidx=None, con=None):
        return cls("Rz", qubit, con, arg, argidx)
