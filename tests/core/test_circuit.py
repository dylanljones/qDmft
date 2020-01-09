# -*- coding: utf-8 -*-
"""
Created on 03 Dec 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import pytest
from numpy.testing import assert_array_equal
from qsim.core.utils import kron, ONE, ZERO
from qsim.core.circuit import Circuit


def test_run_shot():
    c = Circuit(2, 2)
    c.x(0)
    c.mz([0, 1])

    res = c.run_circuit()
    assert_array_equal(res, [-1, 1])

    s0 = kron(ONE, ONE)
    res = c.run_circuit(state=s0)
    assert_array_equal(res, [1, -1])

