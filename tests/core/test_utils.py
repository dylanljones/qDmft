# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: qsim
version: 1.0
"""
from tests.testing import TestCase, main
from qsim.core.utils import *


class TestUtils(TestCase):

    def test_kron(self):
        self.assert_array_equal(kron(np.eye(2), np.eye(2)), np.eye(4))
        self.assert_array_equal(kron([np.eye(2), np.eye(2), np.eye(2)]), np.eye(8))

        a = np.array([[0, 1], [1, 0]])
        aa = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        self.assert_array_equal(kron(a, a), aa)

    def test_get_projector(self):
        self.assert_array_equal(get_projector(ZERO), P0)
        self.assert_array_equal(get_projector(ONE), P1)

        self.assert_array_almost_equal(get_projector(PLUS), 0.5 * np.array([[1, 1], [1, 1]]))
        self.assert_array_almost_equal(get_projector(MINUS), 0.5 * np.array([[1, -1], [-1, 1]]))

        self.assert_array_almost_equal(get_projector(IPLUS), 0.5 * np.array([[1, -1j], [1j, 1]]))
        self.assert_array_almost_equal(get_projector(IMINUS), 0.5 * np.array([[1, 1j], [-1j, 1]]))

    def test_expectation(self):
        self.assertEqual(2, expectation(sx, [1, 1]))
        self.assertEqual(-2, expectation(sx, [-1, 1]))
        self.assertEqual(2, expectation(sy, [1, 1j]))
        self.assertEqual(-2, expectation(sy, [1, -1j]))
        self.assertEqual(1, expectation(sz, [1, 0]))
        self.assertEqual(-1, expectation(sz, [0, 1]))

    def test_binstr(self):
        n = 3
        self.assertEqual("001", binstr(1, n))
        self.assertEqual("011", binstr(3, n))
        self.assertEqual("000", binstr(0, n))

    def test_basis_states(self):
        states = basis_states(3)
        self.assertEqual(3, len(set(states)))

    def test_basis_strings(self):
        strings = ["|00>", "|01>", "|10>", "|11>"]
        self.assertEqual(strings, basis_strings(2))

    def test_to_array(self):
        self.assert_array_equal([1], to_array(1))
        self.assert_array_equal([1, 2], to_array([1, 2]))

    def test_to_list(self):
        self.assert_array_equal([1], to_list(1))
        self.assert_array_equal([1, 2], to_list([1, 2]))

    def test_str_to_list(self):
        self.assertEqual([], str_to_list("a,", dtype=int))
        self.assertEqual([1, 2], str_to_list("1 2", dtype=int))
        self.assertEqual([1, 2], str_to_list("1, 2", dtype=int))
        self.assertEqual([1, 2], str_to_list("1; 2", dtype=int))
        self.assertEqual([1.0, 2.0], str_to_list("1 2", dtype=float))
        self.assertEqual([1.1, 2.2], str_to_list("1.1 2.2", dtype=float))

    def test_get_info(self):
        string = "a=1; b=2.2; c=3; d=[1, 2, 3, 4]; e=5"
        self.assertEqual("1", get_info(string, "a"))
        self.assertEqual("2.2", get_info(string, "b"))
        self.assertEqual("3", get_info(string, "c"))
        self.assertEqual("[1, 2, 3, 4]", get_info(string, "d"))
        self.assertEqual("5", get_info(string, "e"))
        self.assertEqual("", get_info(string, "f"))


if __name__ == "__main__":
    main()
