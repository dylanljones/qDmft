# -*- coding: utf-8 -*-
"""
Created on 20 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
import itertools
from .utils import to_list


class Bit:

    COUNTER = 0

    def __init__(self, index=None, register=None):
        index = Bit.COUNTER if index is None else index
        self.index = index
        self.register = register
        Bit.COUNTER += 1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index})"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.index == other
        else:
            return self.index == other.index and self.register == other.register

    def to_string(self):
        pass


class Qubit(Bit):

    def __init__(self, index=None, register=None):
        super().__init__(index, register)

    def to_string(self):
        reg = f"{self.register.to_string}" if self.register is not None else "None"
        return f"Qubit: index={self.index}, reg={reg}"


class Clbit(Bit):

    def __init__(self, index=None, register=None):
        super().__init__(index, register)

    def to_string(self):
        reg = f"{self.register.to_string}" if self.register is not None else "None"
        return f"Clbit: index={self.index}, reg={reg}"


def init_bits(arg, bit_type, reg=None):
    bits = None
    if isinstance(arg, int):
        bits = [bit_type(i, reg) for i in range(arg)]
    elif isinstance(arg, bit_type):
        arg.register = reg
        bits = [arg]
    elif isinstance(arg, list):
        bits = arg
        for b in bits:
            b.register = reg
    return bits


class Register:
    _id_iter = itertools.count()
    bit_type = Bit

    def __init__(self, arg):
        self.idx = next(self._id_iter)
        self.bits = init_bits(arg, self.bit_type, self)

    @property
    def n(self):
        return len(self.bits)

    @property
    def indices(self):
        return [bit.index for bit in self.bits]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx}, size: {self.n})"

    def __str__(self):
        string = self.__repr__()
        for bit in self.bits:
            string += "\n  -" + str(bit)
        return string

    def list(self, bits):
        if bits is None:
            return None
        bitlist = list()
        for c in to_list(bits):
            if not isinstance(c, self.bit_type):
                c = self.bits[c]
            bitlist.append(c)
        return bitlist

    def __getitem__(self, item):
        if hasattr(item, "__len__"):
            return [self.bits[i] for i in item]
        else:
            return self.bits[item]

    def insert(self, bit, index=None):
        index = self.n if index is None else index
        bit.index = index
        bit.register = self
        self.bits.insert(index, bit)
        for bit in self.bits[index + 1:]:
            bit.index += 1

    def index(self, item):
        return self.bits[item]


class ClRegister(Register):

    bit_type = Clbit

    def __init__(self, arg=0, values=0):
        super().__init__(arg)
        if not hasattr(values, "__len__"):
            values = np.ones(self.n) * values
        self.values = np.asarray(values)

    @classmethod
    def like(cls, register):
        return cls(register.n)


class QuRegister(Register):

    bit_type = Qubit

    def __init__(self, size=0):
        super().__init__(size)
