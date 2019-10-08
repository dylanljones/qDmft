# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
from scitools import Plot, Circle


def kron(*args):
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def basis_states(n):
    return list(range(int(n)))


def binstr(x, n):
    return f"{bin(x)[2:]:0>{n}}"


class AmplitudePlot(Plot):

    def __init__(self, n, lim=1.01):
        super().__init__(create=False)
        self.set_gridspec(n, n)
        self.amps = list()
        for i in range(int(n*n)):
            ax = self.add_gridsubplot(i)
            # Configure subplot
            self.set_limits((-lim, lim), (-lim, lim))
            self.set_ticklabels([], [])
            self.set_equal_aspect()

            circ = Circle((0, 0), radius=1.0, fill=False, color="k", lw=0.5)
            ax.add_artist(circ)
            self.amps.append(ax.plot([0, 1], [0, 0], lw=2)[0])
        self.set_figsize(300, ratio=1)
        self.tight()

    def set_amps(self, amps):
        for i in range(len(amps)):
            amp = amps[i]
            points = np.array([[0, 0], [amp.real, amp.imag]])
            self.amps[i].set_data(*points.T)
