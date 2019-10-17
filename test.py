# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import re
import numpy as np
from qsim import Circuit, Gate
from qsim.visuals import Visualizer, state_block, outer_indices, inner_indices, set_char
from qsim.utils import *
from scitools import Plot


def get_circuit(new=False):
    if new:
        c = Circuit(2)
        c.h(0)
        c.x([0, 1])
        c.h(0)
        c.m()
        c.save("test")
    else:
        c = Circuit.load("test")
    return c


def centered(s1, s2, s3, pad=1):
    width = len(s1) + pad * 2
    return [f"{s1: ^{width}}", f"{s2:-^{width}}", f"{s3: ^{width}}"]


class CircuitString(Visualizer):

    def __init__(self, n, pad=0, offset=0):
        super().__init__(n)
        self.m_line = " " * 5
        self.widths = list([5])
        self.layers = [self.state_layer()]
        self.pad = pad
        self.offset = offset

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

    def add_gate(self, indices, name, pad=0, offset=0):
        if not hasattr(indices, "__len__"):
            indices = [indices]

        w = len(name) + 2
        space = self._gate_line(w)
        name = self._gate_line(w, name)
        line = self._gate_line(w, inner="-")
        edge = "+" + "-" * w + "+"
        width = len(name) + 2 * pad + offset
        r0, r1 = outer_indices(indices, indices)
        inner = inner_indices(self.n, indices, indices)
        if r0 == r1:
            self._add_to_row(r0, centered(edge, name, edge, pad))
            # self.layer[r0] += centered(edge, name, edge, pad)
        else:
            self._add_to_row(r0, centered(edge, name, space, pad))
            self._add_to_row(r1, centered(space, space, edge, pad))
            # self.layer[r0] += centered(edge, name, space, pad)
            # self.layer[r1] += centered(space, space, edge, pad)
            for i, row in enumerate(inner):
                self._add_to_row(row, centered(space, space if row in indices else line, space, pad))
                # self.layer[row] += centered(space, space if row in indices else line, space, pad)
        self.widths[-1] = max(width, self.widths[-1])

    def add_measurement(self, qbits, cbits):
        i = 0
        width = 5
        for q, c in zip(qbits, cbits):
            self.add_gate(q, f"M")
            indices = list(range(self.n))
            indices.remove(q)
            for row in indices:
                char = " " if row < q else "|"
                self._add_to_row(row, self._line(char, width))
                # self.layer[row] = self._line(self.widths[-1])
            i += 1
        self.widths[-1] = i * width

    def add_control_gate(self, gate, pad=0, offset=0):
        idx = gate.qbits
        con = gate.con
        self.add_gate(idx, gate.name.replace("c", ""), pad, offset)
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

    def add(self, inst):
        pad = self.pad
        self.next_layer()
        if inst.name.lower() == "m":
            self.add_measurement(inst.qbits, inst.cbits)
        elif inst.is_controlled:
            self.add_control_gate(inst, pad)
        else:
            name = inst.name
            if inst.arg is not None:
                name += f" ({inst.arg:.2f})"
            for q in inst.qbits:
                self.add_gate([q], name, pad)

    def add_end(self, width=2):
        self.next_layer()
        line = "-" * width + "|"
        space = " " * len(line)
        for i in range(self.n):
            self.layer[i] = [space, line, space]

    def add_layer(self, lines, idx, padding=0):
        i = 0
        width = self.widths[idx]
        for row in self.layers[idx]:
            lines[i + 0] += f"{row[0]: ^{width}}" + " " * padding
            lines[i + 1] += f"{row[1]:-^{width}}" + "-" * padding
            lines[i + 2] += f"{row[2]: ^{width}}" + " " * padding
            i += 3
        return lines

    def build(self, pading=4, wmax=None):
        widths = np.asarray(self.widths) + pading
        header = "".join([f"{i: ^{widths[i]}}" for i in range(1, self.num_layers)])
        string = " " * widths[0] + header + "\n"
        # string = ""
        # string += build_string(self.layers, self.n, self.widths)

        lines = [""] * 3 * self.n
        for i in range(self.num_layers):
            lines = self.add_layer(lines, i, pading)
        if wmax is not None:
            for i, string in enumerate(lines):
                lines[i] = string[:wmax]

        return string + "\n".join(lines)

    def __str__(self):
        return self.build()

    def show(self):
        print(self)


def build_layer(layer):
    lines = list()
    for row in layer:
        lines += row
    return "\n".join(lines)


def main():
    c = get_circuit(True)

    s = CircuitString(c.qbits)
    for inst in c.instructions:
        s.add(inst)
    print(build_layer(s.layer))
    print(s)

if __name__ == "__main__":
    main()
