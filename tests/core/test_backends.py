# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: qsim
version: 1.0
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from qsim.core.utils import *
from qsim.core.register import QuRegister
from qsim.core.backends import StateVector

si, sx, sy, sz = pauli

reg = QuRegister(2)
state = StateVector(reg)


def test_set():
    state.set([0, 0, 1, 0])
    assert_array_equal(state.amp, [0, 0, 1, 0])

    state.set()
    assert_array_equal(state.amp, [1, 0, 0, 0])

    with pytest.raises(ValueError):
        state.set([1, 1, 0, 0])

    with pytest.raises(ValueError):
        state.set([1, 0, 0])


def test_prepare():
    state.prepare([ONE, ZERO])
    assert_array_equal(state.amp, [0, 0, 1, 0])

    state.prepare(ONE, ZERO)
    assert_array_equal(state.amp, [0, 0, 1, 0])

    state.prepare([ONE, ONE])
    assert_array_equal(state.amp, [0, 0, 0, 1])


def test_density_matrix():
    rho = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    state.set(kron([ZERO, ZERO]))
    assert_array_equal(state.density_matrix(), rho)


def test_expectation():
    # <-|x|-> = -1,  <+|x|+> = +1
    op = kron(sx, si)
    state.prepare(MINUS, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == -1.0

    state.prepare(PLUS, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == +1.0

    # <-i|x|-i> = -1,  <i|x|i> = +1
    op = kron(sy, si)
    state.prepare(IMINUS, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == -1.0

    state.prepare(IPLUS, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == +1.0

    # <0|x|0> = +1,  <1|x|1> = -1
    op = kron(sz, si)
    state.prepare(ZERO, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == +1.0

    state.prepare(ONE, ZERO)
    x = state.expectation(op)
    assert pytest.approx(x, 1e-10) == -1.0


def test_project():
    state.prepare(PLUS, PLUS)
    p = state.project(0, P0)
    assert_array_almost_equal(p, [0.5, 0.5, 0, 0], decimal=10)

    p = state.project(0, P1)
    assert_array_almost_equal(p, [0, 0, 0.5, 0.5], decimal=10)

    state.set()
    p = state.project(0, P0)
    assert_array_almost_equal(p, [1, 0, 0, 0], decimal=10)

    p = state.project(0, P1)
    assert_array_almost_equal(p, [0, 0, 0, 0], decimal=10)


def test_apply_unitary():
    state.set()
    state.apply_unitary(kron(sx, si))
    assert_array_equal(state.amp, [0, 0, 1, 0])

    state.apply_unitary(kron(sx, si))
    assert_array_equal(state.amp, [1, 0, 0, 0])

    state.apply_unitary(kron(sx, sx))
    assert_array_equal(state.amp, [0, 0, 0, 1])

    state.apply_unitary(kron(sx, sx))
    assert_array_equal(state.amp, [1, 0, 0, 0])


def test_measure_qubit():
    state.prepare([ZERO, ZERO])
    x = state.measure_qubit(reg[0])
    assert x == 0.

    state.prepare([ONE, ZERO])
    x = state.measure_qubit(reg[0])
    assert x == 1.

    # <-|x|-> = -1,  <+|x|+> = +1
    state.prepare([MINUS, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_X)
    assert x == -1

    state.prepare([PLUS, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_X)
    assert x == +1

    # <-i|x|-i> = -1,  <i|x|i> = +1
    state.prepare([IMINUS, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_Y)
    assert x == -1

    state.prepare([IPLUS, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_Y)
    assert x == +1

    # <0|x|0> = +1,  <1|x|1> = -1
    state.prepare([ZERO, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_Z)
    assert x == +1

    state.prepare([ONE, ZERO])
    x = state.measure_qubit(reg[0], EIGVALS, EV_Z)
    assert x == -1


def test_measure_qubit_post_state():
    state.prepare(PLUS, PLUS)
    val = state.measure_qubit(reg[0], EIGVALS, EV_Z)
    if val == -1:
        expected = np.array([0, 0, 1, 1]) / np.sqrt(2)
    else:
        expected = np.array([1, 1, 0, 0]) / np.sqrt(2)
    assert_array_almost_equal(state.amp, expected)


def test_measure():
    # Single Qubit measurement
    # ------------------------
    state.prepare([ZERO, ZERO])
    x = state.measure(reg[0])[0]
    assert x == 0.

    state.prepare([ONE, ZERO])
    x = state.measure(reg[0])[0]
    assert x == 1.

    # <-|x|-> = -1,  <+|x|+> = +1
    state.prepare([MINUS, ZERO])
    x = state.measure_x(reg[0])[0]
    assert x == -1.

    state.prepare([PLUS, ZERO])
    x = state.measure_x(reg[0])[0]
    assert x == +1

    # <-i|x|-i> = -1,  <i|x|i> = +1
    state.prepare([IMINUS, ZERO])
    x = state.measure_y(reg[0])[0]
    assert x == -1.

    state.prepare([IPLUS, ZERO])
    x = state.measure_y(reg[0])[0]
    assert x == +1.

    # <0|x|0> = +1,  <1|x|1> = -1
    state.prepare([ZERO, ZERO])
    x = state.measure_z(reg[0])[0]
    assert x == +1.

    state.prepare([ONE, ZERO])
    x = state.measure_z(reg[0])[0]
    assert x == -1.

    # Multi Qubit measurement
    # ------------------------
    state.prepare([ZERO, ZERO])
    x = state.measure(reg[0:2])
    assert x == [0, 0]

    state.prepare([ONE, ONE])
    x = state.measure(reg[0:2])
    assert x == [1, 1]

    state.prepare([ZERO, ONE])
    x = state.measure(reg[0:2])
    assert x == [0, 1]

    state.prepare([ONE, ZERO])
    x = state.measure(reg[0:2])
    assert x == [1, 0]

    # <-|x|-> = -1,  <+|x|+> = +1
    state.prepare([MINUS, MINUS])
    x = state.measure_x(reg[0:2])
    assert x == [-1, -1]

    state.prepare([PLUS, PLUS])
    x = state.measure_x(reg[0:2])
    assert x == [+1, +1]

    state.prepare([MINUS, PLUS])
    x = state.measure_x(reg[0:2])
    assert x == [-1, +1]

    state.prepare([PLUS, MINUS])
    x = state.measure_x(reg[0:2])
    assert x == [+1, -1]

    # <-i|y|-i> = -1,  <i|y|i> = +1
    state.prepare([IMINUS, IMINUS])
    x = state.measure_y(reg[0:2])
    assert x == [-1, -1]

    state.prepare([IPLUS, IPLUS])
    x = state.measure_y(reg[0:2])
    assert x == [+1, +1]

    state.prepare([IMINUS, IPLUS])
    x = state.measure_y(reg[0:2])
    assert x == [-1, +1]

    state.prepare([IPLUS, IMINUS])
    x = state.measure_y(reg[0:2])
    assert x == [+1, -1]

    # <0|z|0> = +1,  <1|z|1> = -1
    state.prepare([ZERO, ZERO])
    x = state.measure_z(reg[0:2])
    assert x == [+1, +1]

    state.prepare([ONE, ONE])
    x = state.measure_z(reg[0:2])
    assert x == [-1, -1]

    state.prepare([ZERO, ONE])
    x = state.measure_z(reg[0:2])
    assert x == [+1, -1]

    state.prepare([ONE, ZERO])
    x = state.measure_z(reg[0:2])
    assert x == [-1, +1]
