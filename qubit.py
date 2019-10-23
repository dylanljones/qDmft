# -*- coding: utf-8 -*-
"""
Created on 18 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import os
import numpy as np
from qsim import Circuit

FILE = "test2"


def get_circuit(new=False, file=FILE):
    if new or not os.path.isfile(file):
        print("Saving circuit:", file)
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.m()
        c.save(file)
    else:
        print("Loading circuit:", file)
        c = Circuit.load(file)
    return c


def main():
    # save(FILE)
    c = get_circuit(True)
    c.print()

    c.add_qubit(0)
    # c.print()

    c.print()
    c.run(1000)
    c.res.show_histogram()



if __name__ == "__main__":
    main()
