# -*- coding: utf-8 -*-
"""
Created on 22 Sep 2019
author: Dylan Jones

project: Qsim
version: 1.0
"""
import numpy as np
import scipy.linalg as la
import itertools
import matplotlib.pyplot as plt
from scitools import Plot
from _qsim_old import Gate, Circuit
from _qsim_old.objects import *


def truth_table(c):
    print("Input  Output")
    inputs = list(itertools.product(["0", "1"], repeat=c.n))
    for inp in inputs:
        c.set_state("".join(inp))
        c.run(verbose=False)
        res, p = c.measure()
        res = ", ".join([str(int(x)) for x in res])
        print(", ".join(inp), " ", res)


def teleportation_circuit():
    c = Circuit("00")
    c.add_gate(c.cx(0, 1))
    c.add_gate(c.h(0))
    c.add_measurement([0, 1])
    c.add_gate(c.cx(1, 2))
    c.add_gate(c.cz(0, 2))

    c.run(False)
    c.add_measurement([0, 1])
    print(c.register)
    print(c.measure())


def rz_gate(phi=np.pi):
    arg = 1j * phi / 2
    return np.array([[np.exp(-arg), 0], [0, np.exp(arg)]])


def show_amplitudes(amps):
    n = amps.shape[0]
    fig, axs = plt.subplots(*amps.shape)
    for i in range(n):
        for j in range(n):
            amp = amps[i, j]
            ax = axs[i, j]
            ax.plot([0, 0], [amp.real, amp.imag])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
    plt.show()


def main():
    x = np.pi / 3
    c = Circuit("00")
    g = c.h(0)
    print(g)
    c.add_gate(g)

    # c.add_gate(c.cx(1, 0))
    # c.add_gate(c.rx(0, 1, x))
    # c.add_gate(c.cx(1, 0))

    c.run()
    amps = c.amplitudes()
    print(amps)
    show_amplitudes(amps)


if __name__ == "__main__":
    main()
