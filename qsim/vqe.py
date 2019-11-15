# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
import scipy.linalg as la
from scipy import optimize
from .circuit import Circuit
from .core.utils import expectation


class VqeResult(optimize.OptimizeResult):

    def __init__(self, sol, gs_ref=0):
        super().__init__(x=sol.x, success=sol.success, message=sol.message,
                         nfev=sol.nfev, njev=sol.njev, nit=sol.nit)
        self.gs_ref = gs_ref
        self.gs = sol.fun

    @property
    def error(self):
        return abs(self.gs - self.gs_ref)

    def string(self, d=5, maxvals=10):
        popt_str = ", ".join([f"{x:.{d}}" for x in self.x[:maxvals]])
        if len(self.x) > maxvals:
            popt_str += ", ..."
        lines = list()
        lines.append(f"Message: {self.message}")
        lines.append(f"Evals:   nfev={self.nfev}, njev={self.njev}, nit={self.nit}")
        lines.append(f"Value:   {self.gs_ref:.{d}}")
        lines.append(f"Error:   {self.error:.{d}}")
        lines.append(f"Popt:    {popt_str}")

        line = "-" * (max([len(x) for x in lines]) + 1)
        string = "\n".join(lines)
        return f"{line}\n{string}\n{line}"

    def __str__(self):
        return self.string()


class VqeSolver:

    def __init__(self, ham, num_clbits=None):
        self.ham = None
        self.gs_ref = 0
        self.circuit = None
        self.res = None
        self.init(ham, num_clbits)

    def init(self, ham, num_clbits=None):
        num_qubits = int(np.log2(ham.shape[0]))
        eigvals, eigstates = la.eigh(ham)
        gs = np.min(eigvals)
        self.ham = ham
        self.gs_ref = gs
        self.circuit = Circuit(num_qubits, num_clbits)
        self.res = None

    @property
    def success(self):
        return self.res.success

    @property
    def popt(self):
        return self.res.x

    def __str__(self):
        string = "VQE-Solver:"
        if self.res is not None:
            string += "\n" + str(self.res)
        else:
            string += " Ready!"
        return string

    def set_circuit(self, circuit):
        self.circuit = circuit

    def expectation(self, params):
        self.circuit.set_params(params)
        self.circuit.run_shot()
        return self.circuit.expectation(self.ham)

    def eigval(self):
        if self.success:
            return self.expectation(self.res.x)
        return None

    def solve(self, x0=None, tol=1e-10, verbose=False):
        if verbose:
            print("Optimizing Vqe circuit:")
        if x0 is None:
            x0 = np.random.uniform(0, 2 * np.pi, size=self.circuit.n_params)
        sol = optimize.minimize(self.expectation, x0=x0, tol=tol)
        res = VqeResult(sol, self.gs_ref)
        self.res = res
        if verbose:
            print(res)
        return res

    def save(self, name):
        return self.circuit.save(name + ".circ")


def prepare_ground_state(ham, circuit_config, file="",x0=None, new=False, clbits=1, verbose=True):
    print()
    if file and not new:
        try:
            c = Circuit.load(file)
            if verbose:
                print(f"Circuit: {file} loaded!")
            return c
        except FileNotFoundError:
            print(f"No file {file} found.")
    vqe = VqeSolver(ham, clbits)
    circuit_config(vqe)
    vqe.solve(x0=x0, verbose=verbose)
    if file:
        file = vqe.save(file)
        if verbose:
            print(f"Saving circuit: {file}")
    print()
    return vqe.circuit


def test_vqe(c, ham, dec=5):
    eigvals, eigstates = la.eigh(ham)
    gs_ref = np.min(eigvals)
    c.run_shot()
    gs = c.expectation(ham)
    print(f"Ground state:    {gs_ref:.{dec}}")
    print(f"VQE-Preperation: {gs:.{dec}}")
    print(f"Error:           {abs(gs_ref-gs):.3}")
