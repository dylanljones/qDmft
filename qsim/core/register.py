# -*- coding: utf-8 -*-
"""
Created on 20 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
import itertools


class Bit:

    CUNTER = 0

    def __init__(self, index=None, register=None):
        index = Bit.CUNTER if index is None else index
        self.index = index
        self.register = register
        Bit.CUNTER += 1

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


class Register:
    _id_iter = itertools.count()
    bit_type = None

    def __init__(self, size=0):
        self.idx = next(self._id_iter)
        self.bits = [self.bit_type(i, self) for i in range(size)]

    @property
    def n(self):
        return len(self.bits)

    @property
    def indices(self):
        return [bit.index for bit in self.bits]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx}, size: {self.n})"

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

    def __init__(self, size=0, values=0):
        super().__init__(size)
        if not hasattr(values, "__len__"):
            values = np.ones(size) * values
        self.values = np.asarray(values)

    @classmethod
    def like(cls, register):
        return cls(register.n)


class QuRegister(Register):

    bit_type = Qubit

    def __init__(self, size=0):
        super().__init__(size)

    def __str__(self):
        string = ""



