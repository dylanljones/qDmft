# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import re
import numpy as np
from .utils import to_array
import pickle


class ParameterMap:

    INSTANCE = None

    def __init__(self):
        self.indices = list()
        self.params = list()

    @classmethod
    def instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = cls()
        return cls.INSTANCE

    @property
    def num_params(self):
        return len(self.params)

    @property
    def n(self):
        return len(self.indices)

    @property
    def args(self):
        return [self.get(i) for i in range(self.n)]

    def init(self, *args):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, int):
                args = np.zeros(args)
        self.params = list(args)

    def set_params(self, args):
        args = list(args)
        if len(args) != self.num_params:
            raise ValueError(f"Number of parameters doesnt match: {len(args)}!={self.num_params}")
        self.params = list(args)

    def add_param(self, value):
        self.params.append(value)

    def add(self, value=None, idx=None):
        next_idx = None
        if idx is None:
            if value is not None:
                next_idx = len(self.params)
                self.params.append(value)
        else:
            next_idx = idx
        self.indices.append(next_idx)

    def index(self, item):
        return self.indices[item]

    def __getitem__(self, i):
        return self.params[i]

    def __setitem__(self, i, value):
        self.params[i] = value

    def set(self, item, value):
        idx = self.indices[item]
        if idx is None:
            return
        self.params[idx] = value

    def get(self, item):
        idx = self.indices[item]
        if idx is None:
            return None
        return self.params[idx]

    def __str__(self):
        return f"Params: {self.params}, Indices: {self.indices}"


class BitList:

    def __init__(self, *indices):
        self.indices = None
        self.init(*indices)

    def init(self, *indices):
        if len(indices) == 1:
            indices = indices[0]
        if indices is not None:
            indices = to_array(indices)
        self.indices = indices

    def insert(self, idx):
        if self.indices is not None:
            for i in range(len(self.indices)):
                if self.indices[i] >= idx:
                    self.indices[i] += 1

    def __bool__(self):
        return self.indices is not None

    def __getitem__(self, item):
        return self.indices[item]

    def __setitem__(self, item, value):
        self.indices[item] = value

    def __len__(self):
        return len(self.indices)

    def __str__(self):
        return str(self.indices)


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

        self.qbits = BitList(qubits)
        self.con = BitList(con)
        self.cbits = BitList(clbits)

        self.pmap.add(arg, argidx)

    @classmethod
    def from_string(cls, string, delim="; "):
        args = dict()
        for arg in string.split(delim)[:-1]:
            key, val = arg.split("=")
            args.update({key: val})
        name = args["name"]
        qbits = str_to_list(args["qbits"], int)
        con = str_to_list(args["con"], int)
        cbits = str_to_list(args["cbits"], int)
        arg = float(args["arg"]) if args["arg"] != "None" else None
        argidx = int(args["argidx"]) if args["argidx"] != "None" else None
        if argidx == cls.pmap.num_params:
            arg = arg or 0
            argidx = None
        if name.lower() == "m":
            inst = Measurement(name, qbits=qbits, cbits=cbits)
        else:
            inst = Gate(name, qbits, con=con, arg=arg, argidx=argidx)
        return inst

    @property
    def is_controlled(self):
        return bool(self.con)

    @property
    def num_qbits(self):
        return len(self.qbits)

    @property
    def num_con(self):
        return len(self.con)

    @property
    def num_cbits(self):
        return len(self.cbits)

    @property
    def argidx(self):
        return self.pmap.index(self.idx)

    @property
    def arg(self):
        return self.pmap.get(self.idx)

    def insert_qubit(self, idx):
        self.qbits.insert(idx)
        self.con.insert(idx)

    def _attr_str(self):
        parts = [self.name, f"ID: {self.idx}"]
        if self.qbits:
            parts.append(f"qBits: {self.qbits}")
        if self.con:
            parts.append(f"con: {self.con}")
        if self.cbits:
            parts.append(f"cBits: {self.cbits}")
        if self.arg is not None:
            parts.append(f"Args: {self.arg}")
        return ", ".join(parts)

    def to_dict(self):
        return dict(idx=self.idx, name=self.name, qbits=self.qbits,
                    con=self.con, cbits=self.cbits, arg=self.arg,
                    argidx=self.argidx)

    def to_string(self, delim="; "):
        string = ""
        for key, val in self.to_dict().items():
            string += f"{key}={val}{delim}"
        return string

    def __str__(self):
        return f"{self.TYPE}({self._attr_str()})"

    def set_arg(self, value):
        self.pmap.set(self.idx, value)


class Measurement(Instruction):

    TYPE = "Measurement"

    def __init__(self, name, qbits, cbits=None):
        if cbits is None:
            cbits = qbits
        super().__init__(name, qbits, clbits=cbits)


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

    def to_qasm(self, frmt="3e"):
        string = f"{self.name: <3}"
        if self.is_controlled:
            string += f" q{self.qbits:}"
            string += f", q{self.con}"
        else:
            string += f" q{self.qbits}"
        arg = self.arg
        if arg is not None:
            string += f", {arg:.{frmt}}"
        return string
