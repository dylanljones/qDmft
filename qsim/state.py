# -*- coding: utf-8 -*-
"""
Created on 22 Sep 2019
author: Dylan Jones

project: Qsim
version: 1.0
"""
import random
import numpy as np
import scipy.linalg as la
from .utils import kron

_DECIMALS = 10


class BasisState(int):

    @property
    def bin(self):
        return bin(self)[2:]

    def array(self, n):
        arr = np.zeros(n)
        for i in range(n):
            arr[i] = self.get_bit(n-i-1)
        return arr

    def get_bit(self, i):
        return self >> i & 1

    def zerofill(self, length):
        return f"{self.bin:0>{length}}"

    def __str__(self):
        return self.bin


class State(np.ndarray):

    zero = np.array([[1, 0]]).T
    one = np.array([[0, 1]]).T
    plus = np.array([[1, 1]]).T / np.sqrt(2)
    minus = np.array([[1, -1]]).T / np.sqrt(2)

    p0 = np.dot(zero, zero.T)
    p1 = np.dot(one, one.T)

    def __new__(cls, x, normalize=True, dtype=None):
        if isinstance(x, str):
            x = x.strip(" ")
            states = list()
            for bit in x:
                if bit == "0":
                    states.append(cls.zero)
                elif bit == "1":
                    states.append(cls.one)
                elif bit == "+":
                    states.append(cls.plus)
                elif bit == "-":
                    states.append(cls.minus)
            initial_state = kron(states)
        elif isinstance(x, int):
            initial_state = kron([cls.zero] * x)
        elif isinstance(x, State):
            initial_state = np.asarray(x)
        else:
            initial_state = np.asarray(x)
            if len(initial_state.shape) == 1:
                initial_state = np.asarray([x]).T

        arr = np.asarray(initial_state)
        if normalize:
            arr = arr / la.norm(arr)
        obj = arr.view(cls)
        return obj

    def __setitem__(self, key, value):
        super().__setitem__((key, 0), value)

    def __getitem__(self, item):
        return super().__getitem__((item, 0))

    def __str__(self):
        elements = [f"{x}" for x in self]
        return "State: [" + ", ".join(elements) + "]"

    @property
    def n(self):
        return self.shape[0]

    @property
    def n_qubits(self):
        return int(np.log2(self.n))

    @property
    def norm(self):
        return np.round(la.norm(self), _DECIMALS)


class Register:

    def __init__(self, regs):
        """ Initialize Qubit register

        Parameters
        ----------
        regs: int or str or array_like or State
            Input argument of register. This can be either a
            string or array indicating the initial state or an
            integer defining the number of qubits.
        """
        self.state = State(regs)
        num_qubits = self.state.n_qubits
        self.n = num_qubits
        self.n_states = 2 ** num_qubits
        self.basis = [BasisState(i) for i in range(self.n_states)]

    def basis_string(self, width=7):
        string = ""
        for i in range(self.n_states):
            s = f"|{self.basis[i].zerofill(self.n)}>"
            string += f"{s:^{width}}"
        return string

    def register_string(self, width=7):
        string = ""
        for i in range(self.n_states):
            s = f"{self.state[i]:.2f}"
            string += f"{s:^{width}}"
        return string

    def apply_gate(self, gate):
        self.state = State(np.dot(gate, self.state))

    def probabilities(self, decimals=_DECIMALS):
        return np.round(np.abs(self.state.T)**2, decimals=decimals)[0]

    def qubit_indices(self, index):
        idx0, idx1 = list(), list()
        for i in range(self.n_states):
            if self.basis[i].get_bit(self.n - index - 1) == 0:
                idx0.append(i)
            else:
                idx1.append(i)
        return np.array([idx0, idx1])

    def measure(self, simulate=True, decimals=_DECIMALS):
        probs = self.probabilities(decimals)
        # Choose result based on probabilities
        if simulate:
            res = np.random.choice(np.arange(self.n_states), p=probs)
        else:
            res = np.argmax(probs)
        p = probs[res]
        # Collapse other states and renormalize
        self.state[np.arange(self.n_states)] = 0
        self.state[res] = 1
        self.state /= self.state.norm
        return self.basis[res].array(self.n), p

    def measure_qubit(self, index=0, simulate=True, decimals=_DECIMALS):
        indices = self.qubit_indices(index)
        probs = self.probabilities(decimals)
        probs = np.sum(probs[indices], axis=1)
        # Choose result based on probabilities
        if simulate:
            res = np.random.choice([0, 1], p=probs)
        else:
            res = np.argmax(probs)
        p = probs[res]
        # Collapse other states and renormalize
        other_indices = indices[1-res]
        self.state[other_indices] = 0
        self.state /= self.state.norm
        return res, p

    def __str__(self):
        width = 10
        string = "Basis     " + self.basis_string(width) + "\n"
        string += "Amplitude " + self.register_string(width)
        return string
