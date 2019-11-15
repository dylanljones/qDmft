# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import re
import numpy as np
from .core.utils import to_list, get_bit, kron, str_to_list
from .core.gates import GATE_DICT, single_gate, cgate, X_GATE, Y_GATE, Z_GATE



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
    def n(self):
        return len(self.indices)

    @property
    def num_args(self):
        return len(self.args)

    @property
    def num_params(self):
        return len(self.params)

    @property
    def args(self):
        return [self.get(i) for i in range(self.n)]

    def init(self, *args):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, int):
                args = np.zeros(args)
        self.set(args)

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, key, value):
        self.params[key] = value

    def set(self, args):
        self.params = list(args)

    def add_param(self, value):
        self.params.append(value)

    def link_param(self, idx):
        self.indices.append(idx)

    def add_arg(self, args, idx=None):
        if not hasattr(args, "__len__"):
            args = [args]
        indices = list()
        for i, x in enumerate(args):
            indices.append(len(self.params) if idx is None else idx[i])
            self.params.append(x)
        self.indices.append(indices)

    def add_empty(self):
        self.indices.append(None)

    def add(self, arg=None, idx=None):
        if arg is not None:
            self.add_arg(arg, idx)
        elif idx is not None:
            self.link_param(idx)
        else:
            self.add_empty()

    def get(self, i):
        indices = self.indices[i]
        if indices is None:
            return None
        args = [self.params[i] for i in indices]
        return args

    def __str__(self):
        return f"Params: {self.params}, Indices: {self.indices}"


pmap = ParameterMap.instance()


class Instruction:

    INDEX = 0
    TYPE = "Instruction"
    GATE_DICT = GATE_DICT
    pmap = ParameterMap.instance()

    def __init__(self, name, qubits=None, con=None, clbits=None, n=1, arg=None, argidx=None):
        self.idx = Instruction.INDEX
        Instruction.INDEX += 1
        self.size = n
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
        return self.pmap.indices[self.idx]

    @property
    def args(self):
        return self.pmap.get(self.idx)

    def get_arg(self, i=0):
        return self.args[i] if self.args is not None else None

    def _attr_str(self):
        parts = [self.name, f"ID: {self.idx}"]
        if self.n_qubits:
            parts.append(f"qBits: {self.qu_indices}")
        if self.n_con:
            parts.append(f"con: {self.con_indices}")
        if self.n_clbits:
            parts.append(f"cBits: {self.cl_indices}")
        if self.args is not None:
            parts.append(f"Args: {self.args}")
        return ", ".join(parts)

    def __str__(self):
        return f"{self.TYPE}({self._attr_str()})"

    def to_dict(self):
        return dict(idx=self.idx, name=self.name, qbits=self.qu_indices,
                    con=self.con_indices, cbits=self.cl_indices, arg=self.args,
                    argidx=self.argidx)

    def to_string(self, delim="; "):
        string = ""
        for key, val in self.to_dict().items():
            string += f"{key}={val}{delim}"
        return string

    @classmethod
    def from_string(cls, string, qubit_list, clbit_list, delim="; "):
        qubits, con, clbits = None, None, None
        args = dict()
        for arg in string.split(delim)[:-1]:
            key, val = arg.split("=")
            args.update({key: val})
        name = args["name"]

        qu_indices = str_to_list(args["qbits"], int)
        if qu_indices is not None:
            qubits = [get_bit(qubit_list, idx) for idx in qu_indices]

        con_indices = str_to_list(args["con"], int)
        if con_indices is not None:
            con = [get_bit(qubit_list, idx) for idx in con_indices]

        cl_indices = str_to_list(args["qbits"], int)
        if cl_indices is not None:
            clbits = [get_bit(clbit_list, idx) for idx in cl_indices]

        arg = str_to_list(args["arg"], float) if args["arg"] != "None" else None
        argidx = str_to_list(args["argidx"], int) if args["argidx"] != "None" else None
        if name.lower() == "m":
            inst = Measurement(name, qubits=qubits, clbits=clbits)
        else:
            inst = Gate(name, qubits, con=con, arg=arg, argidx=argidx)
        return inst


class Measurement(Instruction):

    TYPE = "Measurement"

    def __init__(self, name, qubits, clbits=None, basis=None):
        if clbits is None:
            clbits = qubits
        super().__init__(name, qubits, clbits=clbits)
        self.basis = basis

    @classmethod
    def x(cls, qubits, clbits=None):
        return cls("m", qubits, clbits, basis="x")

    @classmethod
    def y(cls, qubits, clbits=None):
        return cls("m", qubits, clbits, basis="y")

    @classmethod
    def z(cls, qubits, clbits=None):
        return cls("m", qubits, clbits, basis="z")

    def basis_operator(self):
        if not self.basis:
            return None

        if self.basis.lower() == "x":
            return X_GATE
        elif self.basis.lower() == "y":
            return Y_GATE
        elif self.basis.lower() == "z":
            return Z_GATE
        else:
            return None


class Gate(Instruction):

    TYPE = "Gate"

    def __init__(self, name, qubits, con=None, arg=None, argidx=None, n=1):
        if hasattr(qubits, "__len__"):
            if arg is not None and not hasattr(arg, "__len__"):
                arg = [arg] * len(qubits)
            if argidx is not None and not hasattr(argidx, "__len__"):
                argidx = [argidx] * len(qubits)
        super().__init__(name, qubits, con=con, n=n, arg=arg, argidx=argidx)
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

    @classmethod
    def xy(cls, qubit1, qubit2, arg=0, argidx=None):
        return cls("XY", [qubit1, qubit2], arg=arg, argidx=argidx, n=2)

    @classmethod
    def add_custom_gate(cls, name, item):
        cls.GATE_DICT.update({name: item})

    @classmethod
    def _get_gatefunc(cls, name):
        func = cls.GATE_DICT.get(name.lower())
        if func is None:
            raise KeyError(f"Gate-function \'{name}\' not in dictionary")
        return func

    def _qubit_gate_matrix(self, idx):
        arg = self.get_arg(idx)
        func = self._get_gatefunc(self.name)
        return func(arg)

    def _build_single_gate(self, n_qubits):
        indices = self.qu_indices
        eye = np.eye(2)
        arrs = list()
        for qbit in range(n_qubits):
            part = eye
            if qbit in indices:
                idx = indices.index(qbit)
                part = self._qubit_gate_matrix(idx)
            arrs.append(part)
        return kron(arrs)

    def build_matrix(self, n_qubits):
        if self.is_controlled:
            name = self.name.replace("c", "")
            gate_func = self._get_gatefunc(name)
            arr = cgate(self.con_indices, self.qu_indices[0], gate_func(self.get_arg()), n_qubits)
        elif self.size > 1:
            gate_func = self._get_gatefunc(self.name)
            arr = gate_func(self.qu_indices, n_qubits, self.get_arg())
        else:
            # arr = self._build_single_gate(n_qubits)
            indices = self.qu_indices
            gate_matrices = list([self._qubit_gate_matrix(i) for i in range(len(indices))])
            arr = single_gate(indices, gate_matrices, n_qubits)
        return arr
