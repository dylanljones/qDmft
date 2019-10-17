# -*- coding: utf-8 -*-
"""
Created on 11 Oct 2019
author: Dylan Jones

project: qDmft
version: 1.0
"""
import numpy as np
from itertools import product
from scitools import Plot, Circle


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


class Visualizer:

    def __init__(self, n):
        self.n = n

    def add(self, inst):
        pass

    def show(self, *args, **kwargs):
        pass


BLOCKWIDTH = 7


def state_block(val=0):
    state = f"|{val}>--"
    space = " " * len(state)
    return [space, state, space]


def gate_block(name, width=11):
    gate = f"| {name} |"
    line = "+" + "-" * (len(gate) - 2) + "+"
    gate = f"{gate:-^{width}}"
    line = f"{line: ^{width}}"
    return [line, gate, line]


def empty_block(width=11):
    empty = " " * width
    line = "-" * width
    return [empty, line, empty]


def add_layer(strings, layer):
    i = 0
    for row in layer:
        for line in row:
            strings[i] += line
            i += 1
    return strings


def build_string(layers, n, width=None):
    strings = [" "] * 3 * n
    for layer in layers:
        strings = add_layer(strings, layer)
    if width is not None:
        for i, line in enumerate(strings):
            strings[i] = line[:width]
    return "\n".join(strings)


def set_char(string, idx, char):
    return string[:idx] + char + string[idx + 1:]


def outer_indices(indices1, indices2):
    combinations = list(product(indices1, indices2))
    idx = int(np.argmax([abs(i - j) for i, j in combinations]))
    return combinations[idx]


def inner_indices(n, indices1, indices2):
    o1, o2 = sorted(outer_indices(indices1, indices2))
    return [i for i in range(n) if o1 < i < o2]


class CircuitString(Visualizer):

    def __init__(self, n, blockwidth=7, maxwidth=None):
        super().__init__(n)
        layer0 = [state_block()] * n
        self.layers = [layer0]
        self.width = blockwidth
        self.maxwidth = maxwidth

    @property
    def layer(self):
        return self.layers[-1]

    @property
    def num_layers(self):
        return len(self.layers)

    def next_layer(self):
        layer = [empty_block(self.width) for _ in range(self.n)]
        self.layers.append(layer)

    def centered(self, s1, s2, s3):
        return [f"{s1: ^{self.width}}", f"{s2:-^{self.width}}", f"{s3: ^{self.width}}"]

    def add_gate(self, indices, name):
        w = len(name) + 2
        name = f"| {name} |"
        edge = "+" + "-" * w + "+"
        line = "|" + "-" * w + "|"
        space = "|" + " " * w + "|"
        r0, r1 = outer_indices(indices, indices)
        inner = inner_indices(self.n, indices, indices)
        if r0 == r1:
            self.layer[r0] = self.centered(edge, name, edge)
            return
        self.layer[r0] = self.centered(edge, name, space)
        self.layer[r1] = self.centered(space, space, edge)
        for i, row in enumerate(inner):
            self.layer[row] = self.centered(space, space if row in indices else line, space)

    def set_char(self, row, line, idx, char):
        string = self.layer[row][line]
        self.layer[row][line] = set_char(string, idx, char)

    def set_chars(self, row, idx, chars):
        for i, char in enumerate(chars):
            self.set_char(row, i, idx, char)

    def add_control_gate(self, gate):
        idx = gate.qbits
        con = gate.con
        self.add_gate(idx, gate.name.replace("c", ""))
        # Connect control qubits
        con_out, idx_out = outer_indices(con, idx)
        x0, x1 = sorted([con_out, idx_out])
        con_in = [row for row in con if x0 < row < x1]
        c = int(self.width / 2)
        # Draw inner sections
        for row in range(self.n):
            if row in con_in:
                self.set_chars(row, c, ["|", "O", "|"])
            elif x0 < row < x1:
                self.set_chars(row, c, ["|", "+", "|"])
        # Draw outer control qubit
        idx = np.sign(idx_out - con_out)
        self.set_char(con_out, 1, c, "O")
        self.set_char(con_out, 1 + idx, c, "|")
        self.set_char(con_out, 1 - idx, c, " ")

    def add(self, instructions):
        self.next_layer()
        if not hasattr(instructions, "__len__"):
            instructions = [instructions]
        for inst in instructions:
            if inst.is_controlled:
                self.add_control_gate(inst)
            else:
                self.add_gate(inst.qbits, inst.name)

    def add_end(self, width=2):
        self.next_layer()
        line = "-" * width + "|"
        space = " " * len(line)
        for i in range(self.n):
            self.layer[i] = [space, line, space]

    def add_endstate(self, vals, decimals=2):
        self.next_layer()
        for i in range(self.n):
            state = f"--|{vals[i]:.{decimals}}>"
            space = " " * len(state)
            self.layer[i] = [space, state, space]

    def build(self):
        header = "".join([f"{i: ^{self.width}}" for i in range(1, self.num_layers)])
        string = " " * 6 + header + "\n"
        string += build_string(self.layers, self.n, self.maxwidth)
        return string

    def __str__(self):
        return self.build()

    def show(self):
        print(self)
