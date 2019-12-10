# -*- coding: utf-8 -*-
"""
Created on 29 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
from qsim.core.register import Bit, Register


def test_bit():
    Bit.COUNTER = 0
    b = Bit()
    assert b.index == 0
    assert b.register is None

    b2 = Bit()
    assert b2.index == 1
    assert b2.register is None


def test_bit_equal():
    b1 = Bit(index=0)
    b2 = Bit(index=1)
    b3 = Bit(index=1)
    reg = Register(1)
    b4 = reg.bits[0]

    assert b2 == b3
    assert b1 != b2
    assert b1 != b4


def test_register():
    reg = Register(3)
    b1, b2, b3 = reg.bits

    assert reg.indices == [0, 1, 2]

    bits = reg.list(0)
    assert bits == [b1]

    bits = reg.list([0, 1])
    assert bits == [b1, b2]
