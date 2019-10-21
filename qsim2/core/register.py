# -*- coding: utf-8 -*-
"""
Created on 20 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np


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

    INDEX = 0
    bit_type = None

    def __init__(self, size=0):
        self.idx = Register.INDEX
        Register.INDEX += 1
        self.bits = [self.bit_type(i, self) for i in range(size)]

    @property
    def n(self):
        return len(self.bits)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx}, size: {self.n})"

    def insert(self, bit, index=None):
        index = self.n if index is None else index
        bit.index = index
        bit.register = self
        self.bits.insert(index, bit)
        for bit in self.bits[index + 1:]:
            bit.index += 1

    def __add__(self, other):
        if isinstance(other, self.bit_type):
            # other.index += self.n
            other.register = self
            self.bits.append(other)
        elif isinstance(other, self.__class__):
            print("Reg")
            for bit in other.bits:
                # bit.index += self.n
                bit.register = self
                self.bits.append(bit)
        return self

    def __radd__(self, other):
        if isinstance(other, self.bit_type):
            # other.index += self.n
            other.register = self
            self.bits.append(other)
        elif isinstance(other, self.__class__):
            print("Reg")
            for bit in other.bits:
                # bit.index += self.n
                bit.register = self
                self.bits.append(bit)
        return self


class QuRegister(Register):

    bit_type = Qubit

    def __init__(self, size=0):
        super().__init__(size)


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
