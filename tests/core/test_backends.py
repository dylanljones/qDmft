# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: qsim
version: 1.0
"""
from tests.testing import TestCase, main
import numpy as np
from qsim.core.utils import *
from qsim.core.register import QuRegister
from qsim.core.backends import StateVector

si, sx, sy, sz = pauli


class TestStateVector(TestCase):

    reg = QuRegister(2)
    s = StateVector(reg)

    def test_set(self):
        self.s.set([0, 0, 1, 0])
        self.assert_array_equal([0, 0, 1, 0], self.s.amp)
        self.s.set()
        self.assert_array_equal([1, 0, 0, 0], self.s.amp)
        self.assertRaises(ValueError, self.s.set, [1, 1, 0, 0])
        self.assertRaises(ValueError, self.s.set, [1, 0, 0])

    def test_density_matrix(self):
        rho = np.array([[1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        self.s.set(kron([ZERO, ZERO]))
        self.assert_array_equal(rho, self.s.density_matrix())

    def test_expectation(self):
        # <-|x|-> = -1,  <+|x|+> = +1
        op = kron(sx, si)
        self.s.prepare(MINUS, ZERO)
        self.assertAlmostEqual(-1.0, self.s.expectation(op))
        self.s.prepare(PLUS, ZERO)
        self.assertAlmostEqual(+1.0, self.s.expectation(op))
        # <-i|x|-i> = -1,  <i|x|i> = +1
        op = kron(sy, si)
        self.s.prepare(IMINUS, ZERO)
        self.assertAlmostEqual(-1.0, self.s.expectation(op))
        self.s.prepare(IPLUS, ZERO)
        self.assertAlmostEqual(+1.0, self.s.expectation(op))
        # <0|x|0> = +1,  <1|x|1> = -1
        op = kron(sz, si)
        self.s.prepare(ZERO, ZERO)
        self.assertAlmostEqual(+1.0, self.s.expectation(op))
        self.s.prepare(ONE, ZERO)
        self.assertAlmostEqual(-1.0, self.s.expectation(op))

    def test_project(self):
        self.s.set(kron(PLUS, PLUS))
        p = self.s.project(0, P0)
        self.assert_array_almost_equal(p, [0.5, 0.5, 0, 0])
        p = self.s.project(0, P1)
        self.assert_array_almost_equal(p, [0, 0, 0.5, 0.5])

        self.s.set()
        p = self.s.project(0, P0)
        self.assert_array_almost_equal(p, [1, 0, 0, 0])
        p = self.s.project(0, P1)
        self.assert_array_almost_equal(p, [0, 0, 0, 0])

    def test_apply_unitary(self):
        self.s.apply_unitary(kron(sx, si))
        self.assert_array_equal([0, 0, 1, 0], self.s.amp)
        self.s.apply_unitary(kron(sx, si))
        self.assert_array_equal([1, 0, 0, 0], self.s.amp)

        self.s.apply_unitary(kron(sx, sx))
        self.assert_array_equal([0, 0, 0, 1], self.s.amp)
        self.s.apply_unitary(kron(sx, sx))
        self.assert_array_equal([1, 0, 0, 0], self.s.amp)

    def test_measure_qubit(self):
        self.s.prepare([ZERO, ZERO])
        self.assertEqual(0, self.s.measure_qubit(self.reg[0]))
        self.s.prepare([ONE, ZERO])
        self.assertEqual(1, self.s.measure_qubit(self.reg[0]))
        # <-|x|-> = -1,  <+|x|+> = +1
        self.s.prepare([MINUS, ZERO])
        self.assertEqual(-1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_X))
        self.s.prepare([PLUS, ZERO])
        self.assertEqual(+1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_X))
        # <-i|x|-i> = -1,  <i|x|i> = +1
        self.s.prepare([IMINUS, ZERO])
        self.assertEqual(-1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_Y))
        self.s.prepare([IPLUS, ZERO])
        self.assertEqual(+1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_Y))
        # <0|x|0> = +1,  <1|x|1> = -1
        self.s.prepare([ZERO, ZERO])
        self.assertEqual(+1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_Z))
        self.s.prepare([ONE, ZERO])
        self.assertEqual(-1, self.s.measure_qubit(self.reg[0], EIGVALS, EV_Z))

        amp = kron(PLUS, PLUS)
        amp /= np.linalg.norm(amp)
        self.s.set(amp)
        val = self.s.measure_qubit(self.reg[0], EIGVALS, EV_Z)
        if val == -1:
            expected = np.array([0, 0, 1, 1]) / np.sqrt(2)
        else:
            expected = np.array([1, 1, 0, 0]) / np.sqrt(2)
        self.assert_array_almost_equal(expected, self.s.amp)

    def test_measure(self):
        # Single Qubit measurement
        # ------------------------
        self.s.prepare([ZERO, ZERO])
        self.assertEqual(0, self.s.measure(self.reg[0])[0])
        self.s.prepare([ONE, ZERO])
        self.assertEqual(1, self.s.measure(self.reg[0])[0])
        # <-|x|-> = -1,  <+|x|+> = +1
        self.s.prepare([MINUS, ZERO])
        self.assertEqual(-1, self.s.measure_x(self.reg[0])[0])
        self.s.prepare([PLUS, ZERO])
        self.assertEqual(+1, self.s.measure_x(self.reg[0])[0])
        # <-i|x|-i> = -1,  <i|x|i> = +1
        self.s.prepare([IMINUS, ZERO])
        self.assertEqual(-1, self.s.measure_y(self.reg[0])[0])
        self.s.prepare([IPLUS, ZERO])
        self.assertEqual(+1, self.s.measure_y(self.reg[0])[0])
        # <0|x|0> = +1,  <1|x|1> = -1
        self.s.prepare([ZERO, ZERO])
        self.assertEqual(+1, self.s.measure_z(self.reg[0])[0])
        self.s.prepare([ONE, ZERO])
        self.assertEqual(-1, self.s.measure_z(self.reg[0])[0])

        # Multi Qubit measurement
        # ------------------------
        self.s.prepare([ZERO, ZERO])
        self.assertEqual([0, 0], self.s.measure(self.reg[0:2]))
        self.s.prepare([ONE, ONE])
        self.assertEqual([1, 1], self.s.measure(self.reg[0:2]))
        self.s.prepare([ZERO, ONE])
        self.assertEqual([0, 1], self.s.measure(self.reg[0:2]))
        self.s.prepare([ONE, ZERO])
        self.assertEqual([1, 0], self.s.measure(self.reg[0:2]))
        # <-|x|-> = -1,  <+|x|+> = +1
        self.s.prepare([MINUS, MINUS])
        self.assertEqual([-1, -1], self.s.measure_x(self.reg[0:2]))
        self.s.prepare([PLUS, PLUS])
        self.assertEqual([+1, +1], self.s.measure_x(self.reg[0:2]))
        self.s.prepare([MINUS, PLUS])
        self.assertEqual([-1, +1], self.s.measure_x(self.reg[0:2]))
        self.s.prepare([PLUS, MINUS])
        self.assertEqual([+1, -1], self.s.measure_x(self.reg[0:2]))
        # <-i|x|-i> = -1,  <i|x|i> = +1
        self.s.prepare([IMINUS, IMINUS])
        self.assertEqual([-1, -1], self.s.measure_y(self.reg[0:2]))
        self.s.prepare([IPLUS, IPLUS])
        self.assertEqual([+1, +1], self.s.measure_y(self.reg[0:2]))
        self.s.prepare([IMINUS, IPLUS])
        self.assertEqual([-1, +1], self.s.measure_y(self.reg[0:2]))
        self.s.prepare([IPLUS, IMINUS])
        self.assertEqual([+1, -1], self.s.measure_y(self.reg[0:2]))
        # <0|x|0> = +1,  <1|x|1> = -1
        self.s.prepare([ZERO, ZERO])
        self.assertEqual([+1, +1], self.s.measure_z(self.reg[0:2]))
        self.s.prepare([ONE, ONE])
        self.assertEqual([-1, -1], self.s.measure_z(self.reg[0:2]))
        self.s.prepare([ZERO, ONE])
        self.assertEqual([+1, -1], self.s.measure_z(self.reg[0:2]))
        self.s.prepare([ONE, ZERO])
        self.assertEqual([-1, +1], self.s.measure_z(self.reg[0:2]))


if __name__ == "__main__":
    main()
