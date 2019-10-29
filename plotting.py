# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import itertools
from qsim import QuRegister, Qubit, StateVector, Circuit, Gate, kron, pauli
from qsim.core.gates import *
from scitools.plotting import Plot, Colors, legend_patch, hex_to_rgb, Rectangle, RcParams

si, sx, sy, sz = pauli


def project(amp, parts):
    proj = kron(parts)
    return np.dot(proj, amp)


def init_bar_plot(n, labels=None, grid="y", scale=True):
    x_ticks = np.arange(n)
    xlim = -0.5, n-0.5
    ylim = (0, 1.05) if scale is False else None
    plot = Plot(xlim=xlim, ylim=ylim)
    plot.grid(axis=grid)
    plot.set_xticks(x_ticks, labels)
    return plot


def plot_bars(plot, values, col1, col2):
    col1 = hex_to_rgb(col1) / 255
    col2 = hex_to_rgb(col2) / 255
    col = np.array([col2, col1])

    amp = np.abs(np.square(values))
    indices = np.sign(values).real
    indices[indices == 0] = 1
    indices = ((indices + 1) / 2).astype("int")
    colors = [col[i] for i in indices]
    plot.bar(np.arange(len(values)), amp, color=colors)


class GatePatch(Rectangle):

    def __init__(self, pos, name, width=0.3, height=0.3, *args, **kwargs):
        self.pos = pos[0] + width / 2, pos[1] + height / 2
        super().__init__(pos, width, height, *args, **kwargs)
        self.name = name

    def add(self, plot):
        plot.ax.add_artist(self)
        plot.text(self.pos, self.name)


class CircuitPlot(Plot):

    def __init__(self, n_qubits, row_size=0.5):
        self.n_qubits = n_qubits
        self.row_size = row_size
        self.rows = np.arange(n_qubits) * row_size
        self.cols = 0

        margin = row_size / 2
        super().__init__(ylim=(-margin, np.max(self.rows) + margin))
        # RcParams.set_fontsize(12)
        self.set_equal_aspect()
        self.invert_yaxis()
        self.set_yticks(self.rows, [f"|q{i + 1}>" for i in range(self.n_qubits)])
        for r in self.rows:
            self.text((-0.5, r), "|0>")
            self.plot([-0.4, 100], [r, r], color="k", lw=1, zorder=-1)

        self.scale()

    def scale(self):
        self.set_limits(xlim=(-0.7, self.cols + 1))

    def draw_gate(self, x, idx0, idx1, text=""):
        r0, r1 = self.rows[idx0], self.rows[idx1]

        width = 0.5
        height = r1 - r0 + self.row_size - 0.1
        pos = x, -self.row_size / 2 + 0.05
        self.ax.add_artist(Rectangle(pos, width, height))
        center = pos[0] + width/2, pos[1] + height/2
        self.text(center, text)


def main():
    c = Circuit(2)
    c.h()
    c.cx(0, 1)
    c.y(0)
    # c.cx(0, 1)
    n = c.n_qubits
    ratio = n / len(c.instructions)

    height = n / 2

    plot = CircuitPlot(c.n_qubits)
    plot.draw_gate(0, 0, 0, "X")
    plot.draw_gate(0, 1, 1, "X")
    # GatePatch((0, 0.25), "X").add(plot)
    # GatePatch((1, 0.5), "X", height=0.7).add(plot)

    # plot.ax.add_artist(Rectangle((pos[0]-0.25, pos[0]-0.25), width=0.5, height=0.5))
    # plot.text(pos, s=label)

    plot.show()


    return
    c.run()
    print(c.backend.norm)
    c.print()
    s = c.backend
    col1, col2 = Colors.bblue, Colors.cblue

    plot = init_bar_plot(s.n, s.basis.labels, scale=False)
    plot_bars(plot, s.amp, col1, col2)
    handles = [legend_patch("Positive", color=col1),
               legend_patch("Negative", color=col2)]
    plot.legend(handles=handles)
    # plot.bar(range(4), s.probabilities() * np.sign(s.amp))
    plot.show()






if __name__ == "__main__":
    main()
