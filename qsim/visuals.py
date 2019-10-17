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


def centered(s1, s2, s3, pad=1):
    width = len(s1) + pad * 2
    return [f"{s1: ^{width}}", f"{s2:-^{width}}", f"{s3: ^{width}}"]


class CircuitString(Visualizer):

    def __init__(self, n, padding=1):
        super().__init__(n)
        self.m_line = " " * 5
        self.widths = list([5])
        self.layers = [self.state_layer()]
        self.padding = padding

    @property
    def layer(self):
        return self.layers[-1]

    @property
    def idx(self):
        return self.num_layers - 1

    @property
    def num_layers(self):
        return len(self.layers)

    def state_layer(self, vals=None):
        if vals is None:
            vals = np.zeros(self.n, "int")
        layer0 = list()
        w = 0
        for val in vals:
            state = f"|{val}>--"
            space = " " * len(state)
            layer0.append([space, state, space])
            w = max(w, len(state))
        return layer0

    @staticmethod
    def _empty(w=1):
        empty = "" # " " * w
        line = "" # "-" * w
        return [empty, line, empty]

    @staticmethod
    def _line(char=" ", w=1):
        empty = f"{char: ^{w}}"
        line = "-" * w
        return [empty, line, empty]

    @staticmethod
    def _gate_line(w, text="", inner=" "):
        return f"|{text:{inner}^{w}}|"

    def _add_to_row(self, row, strings):
        for i in range(len(strings)):
            self.layer[row][i] += strings[i]

    def next_layer(self):
        self.widths.append(0)
        layer = [self._empty() for _ in range(self.n)]
        self.layers.append(layer)

    def add_gate(self, indices, name, pad=0):
        if not hasattr(indices, "__len__"):
            indices = [indices]

        w = len(name) + 2
        space = self._gate_line(w)
        name = self._gate_line(w, name)
        line = self._gate_line(w, inner="-")
        edge = "+" + "-" * w + "+"
        width = len(name) + 2 * pad
        r0, r1 = outer_indices(indices, indices)
        inner = inner_indices(self.n, indices, indices)
        if r0 == r1:
            # self._add_to_row(r0, centered(edge, name, edge, pad))
            self.layer[r0] = centered(edge, name, edge, pad)
        else:
            # self._add_to_row(r0, centered(edge, name, space, pad))
            # self._add_to_row(r1, centered(space, space, edge, pad))
            self.layer[r0] = centered(edge, name, space, pad)
            self.layer[r1] = centered(space, space, edge, pad)
            for i, row in enumerate(inner):
                # self._add_to_row(row, centered(space, space if row in indices else line, space, pad))
                self.layer[row] = centered(space, space if row in indices else line, space, pad)
        self.widths[-1] = max(width, self.widths[-1])

    def add_control_gate(self, gate, pad=0):
        idx = gate.qbits
        con = gate.con
        self.add_gate(idx, gate.name.replace("c", ""), pad)
        # Connect control qubits
        con_out, idx_out = outer_indices(con, idx)
        x0, x1 = sorted([con_out, idx_out])
        con_in = [row for row in con if x0 < row < x1]
        con_row = ["|", "O", "|"]
        cross_row = ["|", "+", "|"]
        # Draw inner sections
        for row in range(self.n):
            if row in con_in:
                self.layer[row] = con_row
            elif x0 < row < x1:
                self.layer[row] = cross_row
        # Draw outer control qubit
        idx = np.sign(idx_out - con_out)
        outer = ["O"] * 3
        outer[1 + idx] = "|"
        outer[1 - idx] = " "
        self.layer[con_out] = outer

    def add_measurement(self, qbits, cbits):
        for q, c in zip(qbits, cbits):
            self.add_gate(q, f"M {c}")

    def add(self, inst, padding=0):
        self.next_layer()
        if inst.name.lower() == "m":
            self.add_measurement(inst.qbits, inst.cbits)
        elif inst.is_controlled:
            self.add_control_gate(inst, padding)
        else:
            name = inst.name
            if inst.arg is not None:
                name += f" ({inst.arg:.1f})"
            for q in inst.qbits:
                self.add_gate([q], name, padding)

    def add_end(self, width=2):
        self.next_layer()
        line = "-" * width + "|"
        space = " " * len(line)
        for i in range(self.n):
            self.layer[i] = [space, line, space]

    def add_layer(self, lines, idx, padding=0):
        i = 0
        width = self.widths[idx]
        space_pad, line_pad = " " * padding, "-" * padding
        for row in self.layers[idx]:
            if idx == 0:
                lines[i + 0] += f"{row[0]: ^{width}}"
                lines[i + 1] += f"{row[1]:-^{width}}"
                lines[i + 2] += f"{row[2]: ^{width}}"
            else:
                lines[i + 0] += f"{row[0]: ^{width + 2 *padding}}"
                lines[i + 1] += f"{row[1]:-^{width + 2 * padding}}"
                lines[i + 2] += f"{row[2]: ^{width + 2 * padding}}"
            i += 3
        return lines

    def build(self, padding=None, wmax=None):
        pad = padding if padding is not None else self.padding
        circ_lines = [""] * 3 * self.n
        for i in range(self.num_layers):
            circ_lines = self.add_layer(circ_lines, i, pad)
        if wmax is not None:
            for i, string in enumerate(circ_lines):
                circ_lines[i] = string[:wmax]

        widths = np.asarray(self.widths)
        widths[1:] += 2 * pad
        header = "".join([f"{i: ^{widths[i]}}" for i in range(1, self.num_layers)])
        string = " " * widths[0] + header + "\n"
        return string + "\n".join(circ_lines)

    def __str__(self):
        return self.build()

    def show(self):
        print(self)
