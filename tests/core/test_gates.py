# -*- coding: utf-8 -*-
"""
Created on 22 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from qsim.core.gates import *


def test_rx():
    g = rx_gate(0)
    assert_array_almost_equal(g, np.array([[1, 0], [0, 1]]), decimal=10)

    g = rx_gate(1*np.pi)
    assert_array_almost_equal(g, np.array([[0, -1j], [-1j, 0]]), decimal=10)

    g = rx_gate(2*np.pi)
    assert_array_almost_equal(g, np.array([[-1, 0], [0, -1]]), decimal=10)

    g = rx_gate(3*np.pi)
    assert_array_almost_equal(g, np.array([[0, 1j], [1j, 0]]), decimal=10)


def test_ry():
    g = ry_gate(0)
    assert_array_almost_equal(g, np.array([[1, 0], [0, 1]]), decimal=10)

    g = ry_gate(1 * np.pi)
    assert_array_almost_equal(g, np.array([[0, -1], [+1, 0]]), decimal=10)

    g = ry_gate(2 * np.pi)
    assert_array_almost_equal(g, np.array([[-1, 0], [0, -1]]), decimal=10)

    g = ry_gate(3 * np.pi)
    assert_array_almost_equal(g, np.array([[0, +1], [-1, 0]]), decimal=10)


def test_rz():
    g = rz_gate(0)
    assert_array_almost_equal(g, np.array([[1, 0], [0, 1]]), decimal=10)

    g = rz_gate(1 * np.pi)
    assert_array_almost_equal(g, np.array([[-1j, 0], [0, +1j]]), decimal=10)

    g = rz_gate(2 * np.pi)
    assert_array_almost_equal(g, np.array([[-1, 0], [0, -1]]), decimal=10)

    g = rz_gate(3 * np.pi)
    assert_array_almost_equal(g, np.array([[+1j, 0], [0, -1j]]), decimal=10)


def test_single_gate():
    g = single_gate(0, z_gate(), 2)
    z1 = np.diag([1, 1, -1, -1])
    assert_array_equal(g, z1)

    g = single_gate([0, 1], [Z_GATE, Z_GATE])
    z12 = np.diag([1, -1, -1, 1])
    assert_array_equal(g, z12)

    g = single_gate([0, 1], [X_GATE, Y_GATE])
    x1y2 = 1j * np.array([[0,  0,  0, -1],
                          [0,  0, +1,  0],
                          [0, -1,  0,  0],
                          [1,  0,  0,  0]])
    assert_array_equal(g, x1y2)

    g = single_gate(0, Z_GATE, 3)
    z1 = np.diag([1, 1, 1, 1, -1, -1, -1, -1])
    assert_array_equal(g, z1)

    g = single_gate([0, 1], [Z_GATE, Z_GATE], 3)
    z12 = np.diag([1, 1, -1, -1, -1, -1, 1, 1])
    assert_array_equal(g, z12)

    g = single_gate([0, 2], [Z_GATE, Z_GATE], 3)
    z13 = np.diag([1, -1, 1, -1, -1, 1, -1, 1])
    assert_array_equal(g, z13)


def test_cgate():
    g = cgate(0, 1, X_GATE, 2)
    cnot1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
    assert_array_equal(g, cnot1)

    g = cgate(0, 1, X_GATE, 2, trigger=0)
    cnot0 = np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    assert_array_equal(g, cnot0)

    g = cgate(1, 0, X_GATE, 2)
    notc1 = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0]])
    assert_array_equal(g, notc1)

    g = cgate(1, 0, X_GATE, 2, trigger=0)
    notc0 = np.array([[0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]])
    assert_array_equal(g, notc0)


def test_xy_gate():
    g = xy_gatefunc([0, 1], 2, 0)
    xy = np.eye(4)
    assert_array_equal(g, xy)

    g = xy_gatefunc([0, 1], 2, 0.25 * np.pi)
    xy = np.array([[1,   0,   0, 0],
                   [0,   0, -1j, 0],
                   [0, -1j,   0, 0],
                   [0,   0,   0, 1]])
    assert_array_almost_equal(g, xy, decimal=10)

    g = xy_gatefunc([0, 1], 2, 0.5 * np.pi)
    xy = np.array([[1,  0,  0, 0],
                   [0, -1,  0, 0],
                   [0,  0, -1, 0],
                   [0,  0,  0, 1]])
    assert_array_almost_equal(g, xy, decimal=10)

    g = xy_gatefunc([0, 1], 2, 0.75 * np.pi)
    xy = np.array([[1,  0,  0, 0],
                   [0,  0, 1j, 0],
                   [0, 1j,  0, 0],
                   [0,  0,  0, 1]])
    assert_array_almost_equal(g, xy, decimal=10)


def test_b_gate():
    g = b_gatefunc([0, 1], 2, 0)
    b = np.eye(4)
    assert_array_almost_equal(g, b, decimal=10)

    g = b_gatefunc([0, 1], 2, 0.5 * np.pi)
    b = 1j * np.diag([-1, 1, 1, -1])
    assert_array_almost_equal(g, b, decimal=10)

    g = b_gatefunc([0, 1], 2, np.pi)
    b = - np.eye(4)
    assert_array_almost_equal(g, b, decimal=10)

    g = b_gatefunc([0, 1], 2, 1.5*np.pi)
    b = 1j * np.diag([1, -1, -1, 1])
    assert_array_almost_equal(g, b, decimal=10)
