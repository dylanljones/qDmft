# -*- coding: utf-8 -*-
"""
Created on 25 Nov 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import unittest
from unittest import main
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestCase(unittest.TestCase):

    @staticmethod
    def assert_array_equal(x, y):
        assert_array_equal(x, y)

    @staticmethod
    def assert_array_almost_equal(x, y, decimals=15):
        assert_array_almost_equal(x, y, decimals)
