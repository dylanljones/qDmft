# -*- coding: utf-8 -*-
"""
Created on 03 Dec 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
from qsim.core.instruction import ParameterMap, Instruction, Measurement, Gate


def test_parameter_map():
    pmap = ParameterMap()
    pmap.add(100)
    pmap.add(200)
    pmap.add()
    pmap.add(idx=[0])
    assert pmap.params == [100, 200]
    assert pmap.indices == [[0], [1], None, [0]]
    assert pmap.args == [[100], [200], None, [100]]

    pmap[0] = 50
    assert pmap.params == [50, 200]
    assert pmap.args == [[50], [200], None, [50]]
