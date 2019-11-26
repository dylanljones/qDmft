# -*- coding: utf-8 -*-
"""
Created on 22 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
from tests.testing import TestCase, main
from qsim.core.gates import *


class TestGates(TestCase):

    def test_rx(self):
        self.assert_array_almost_equal(rx_gate(0), np.array([[1, 0], [0, 1]]))
        self.assert_array_almost_equal(rx_gate(1*np.pi), np.array([[0, -1j], [-1j, 0]]))
        self.assert_array_almost_equal(rx_gate(2*np.pi), np.array([[-1, 0], [0, -1]]))
        self.assert_array_almost_equal(rx_gate(3*np.pi), np.array([[0, 1j], [1j, 0]]))

    def test_ry(self):
        self.assert_array_almost_equal(ry_gate(0), np.array([[1, 0], [0, 1]]))
        self.assert_array_almost_equal(ry_gate(1*np.pi), np.array([[0, -1], [+1, 0]]))
        self.assert_array_almost_equal(ry_gate(2*np.pi), np.array([[-1, 0], [0, -1]]))
        self.assert_array_almost_equal(ry_gate(3*np.pi), np.array([[0, +1], [-1, 0]]))

    def test_rz(self):
        self.assert_array_almost_equal(rz_gate(0), np.array([[1, 0], [0, 1]]))
        self.assert_array_almost_equal(rz_gate(1*np.pi), np.array([[-1j, 0], [0, +1j]]))
        self.assert_array_almost_equal(rz_gate(2*np.pi), np.array([[-1, 0], [0, -1]]))
        self.assert_array_almost_equal(rz_gate(3*np.pi), np.array([[+1j, 0], [0, -1j]]))

    def test_single_gate(self):
        z1 = np.diag([1, 1, -1, -1])
        self.assert_array_equal(single_gate(0, z_gate(), 2), z1)

        z12 = np.diag([1, -1, -1, 1])
        self.assert_array_equal(single_gate([0, 1], [Z_GATE, Z_GATE]), z12)

        x1y2 = 1j * np.array([[0,  0,  0, -1],
                              [0,  0, +1,  0],
                              [0, -1,  0,  0],
                              [1,  0,  0,  0]])
        self.assert_array_equal(single_gate([0, 1], [X_GATE, Y_GATE]), x1y2)

        z1 = np.diag([1, 1, 1, 1, -1, -1, -1, -1])
        self.assert_array_equal(single_gate(0, Z_GATE, 3), z1)

        z12 = np.diag([1, 1, -1, -1, -1, -1, 1, 1])
        self.assert_array_equal(single_gate([0, 1], [Z_GATE, Z_GATE], 3), z12)

        z13 = np.diag([1, -1, 1, -1, -1, 1, -1, 1])
        self.assert_array_equal(single_gate([0, 2], [Z_GATE, Z_GATE], 3), z13)

    def test_cgate(self):
        cnot1 = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]])
        self.assert_array_equal(cgate(0, 1, X_GATE, 2), cnot1)

        cnot0 = np.array([[0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.assert_array_equal(cgate(0, 1, X_GATE, 2, trigger=0), cnot0)

        notc1 = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0]])
        self.assert_array_equal(cgate(1, 0, X_GATE, 2), notc1)

        notc0 = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]])
        self.assert_array_equal(cgate(1, 0, X_GATE, 2, trigger=0), notc0)

    def test_xy_gate(self):
        xy = xy_gatefunc([0, 1], 2, 0)
        self.assert_array_equal(np.eye(4), xy)

        xy = xy_gatefunc([0, 1], 2, 0.25 * np.pi)
        res = np.array([[1,   0,   0, 0],
                        [0,   0, -1j, 0],
                        [0, -1j,   0, 0],
                        [0,   0,   0, 1]])
        self.assert_array_almost_equal(res, xy, decimals=10)

        xy = xy_gatefunc([0, 1], 2, 0.5 * np.pi)
        res = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])
        self.assert_array_almost_equal(res, xy, decimals=10)

        xy = xy_gatefunc([0, 1], 2, 0.75 * np.pi)
        res = np.array([[1,  0,  0, 0],
                        [0,  0, 1j, 0],
                        [0, 1j,  0, 0],
                        [0,  0,  0, 1]])
        self.assert_array_almost_equal(res, xy, decimals=10)

    def test_b_gate(self):
        b = np.eye(4)
        self.assert_array_almost_equal(b_gatefunc([0, 1], 2, 0), b)
        b = 1j * np.diag([-1, 1, 1, -1])
        self.assert_array_almost_equal(b_gatefunc([0, 1], 2, 0.5*np.pi), b)
        b = - np.eye(4)
        self.assert_array_almost_equal(b_gatefunc([0, 1], 2, np.pi), b)
        b = 1j * np.diag([1, -1, -1, 1])
        self.assert_array_almost_equal(b_gatefunc([0, 1], 2, 1.5*np.pi), b)


if __name__ == "__main__":
    main()
