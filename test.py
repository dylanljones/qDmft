# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from qsim import *
from scitools import Plot


def main():
    c = Circuit(2, 1)
    c.ry(0, np.pi/3)
    c.h(0)
    c.m(0)
    res = c.run(100)
    print(res.mean()[0])
    res.show_histogram()


if __name__ == "__main__":
    main()
