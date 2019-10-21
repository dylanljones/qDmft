# -*- coding: utf-8 -*-
"""
Created on 18 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from qsim2 import Circuit


def main():
    c = Circuit(2)
    c.x(0)
    c.rx(0, np.pi)
    c.m()
    print(c.to_string())


if __name__ == "__main__":
    main()
