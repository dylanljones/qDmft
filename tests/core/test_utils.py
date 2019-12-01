# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from qsim.core.utils import *


def test_kron():
    assert_array_equal(kron(np.eye(2), np.eye(2)), np.eye(4))
    assert_array_equal(kron([np.eye(2), np.eye(2), np.eye(2)]), np.eye(8))

    a = np.array([[0, 1], [1, 0]])
    aa = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(kron(a, a), aa)


def test_get_projector():
    assert_array_equal(get_projector(ZERO), P0)
    assert_array_equal(get_projector(ONE), P1)

    assert_array_almost_equal(get_projector(PLUS), 0.5 * np.array([[1, 1], [1, 1]]), decimal=10)
    assert_array_almost_equal(get_projector(MINUS), 0.5 * np.array([[1, -1], [-1, 1]]), decimal=10)

    assert_array_almost_equal(get_projector(IPLUS), 0.5 * np.array([[1, -1j], [1j, 1]]), decimal=10)
    assert_array_almost_equal(get_projector(IMINUS), 0.5 * np.array([[1, 1j], [-1j, 1]]), decimal=10)


def test_expectation():
    x = expectation(sx, [1, 1])
    assert x == 2

    x = expectation(sx, [-1, 1])
    assert x == -2

    x = expectation(sy, [1, 1j])
    assert x == 2

    x = expectation(sy, [1, -1j])
    assert x == -2

    x = expectation(sz, [1, 0])
    assert x == 1

    x = expectation(sz, [0, 1])
    assert x == -1


def test_binstr():
    n = 3
    s = binstr(1, n)
    assert s == "001"

    s = binstr(3, n)
    assert s == "011"

    s = binstr(0, n)
    assert s == "000"


def test_basisstates():
    states = basis_states(3)
    assert len(states) == 3


def test_basis_strings():
    x = basis_strings(2)
    assert x == ["|00>", "|01>", "|10>", "|11>"]


def test_to_array():
    assert_array_equal(to_array(1), [1])
    assert_array_equal(to_array([1, 2]), [1, 2])


def test_to_list():
    li = to_list(1)
    assert li == [1]

    li = to_list([1, 2])
    assert li == [1, 2]


def test_str_to_list():
    li = str_to_list("a, ", dtype=int)
    assert li == []

    li = str_to_list("1 2", dtype=int)
    assert li == [1, 2]

    li = str_to_list("1, 2", dtype=int)
    assert li == [1, 2]

    li = str_to_list("1; 2", dtype=int)
    assert li == [1, 2]

    li = str_to_list("1 2", dtype=float)
    assert li == [1.0, 2.0]

    li = str_to_list("1.1 2.2", dtype=float)
    assert li == [1.1, 2.2]


def test_get_info():
    string = "a=1; b=2.2; c=3; d=[1, 2, 3, 4]; e=5"

    x = get_info(string, "a")
    assert x == "1"

    x = get_info(string, "b")
    assert x == "2.2"

    x = get_info(string, "c")
    assert x == "3"

    x = get_info(string, "d")
    assert x == "[1, 2, 3, 4]"

    x = get_info(string, "e")
    assert x == "5"

    x = get_info(string, "f")
    assert x == ""
