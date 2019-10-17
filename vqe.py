# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qDmft
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from scipy import optimize
from scitools import Plot
from qsim import kron, pauli
from qsim.circuit import Circuit

si, sx, sy, sz = pauli


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
        string = ""
        string += f"Message: {self.message}\n"
        string += f"Evals:   nfev={self.nfev}, njev={self.njev}, nit={self.nit}\n"
        string += f"Value:   {self.gs_ref:.{d}}\n"
        string += f"Error:   {self.error:.{d}}\n"
        string += f"Popt:    {popt_str}\n"
        return string

    def __str__(self):
        return self.string()


def expectation(operator, state):
    return np.dot(state, np.dot(operator, state).T).real


class VqeSolver:

    def __init__(self, ham, circuit=None):
        self.ham = ham
        eigvals, eigstates = la.eigh(ham)
        self.gs_ref = np.min(eigvals)
        self.circuit = circuit
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
        return expectation(self.ham, self.circuit.state())

    def eigval(self):
        if self.success:
            return self.expectation(self.res.x)
        return None

    def solve(self, x0=None, tol=1e-5):
        if x0 is None:
            x0 = np.random.uniform(0, 2 * np.pi, size=self.circuit.num_params)
        sol = optimize.minimize(self.expectation, x0=x0, tol=tol)
        res = VqeResult(sol, self.gs_ref)
        self.res = res
        return res

    def save(self, name):
        self.circuit.save(name + ".circ")


def hamiltonian(u=4, v=1, eps_bath=2, mu=2):
    h1 = u / 2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    h2 = mu * (kron(sz, si, si, si) + kron(si, si, sz, si))
    h3 = - eps_bath * (kron(si, sz, si, si) + kron(si, si, si, sz))
    h4 = v * (kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(si, si, sx, sx) + kron(si, si, sy, sy))
    return 1/2 * (h1 + h2 + h3 + h4)


def circuit(depth=2):
    c = Circuit(4)
    c.h(0)
    for i in range(depth):
        c.cx(0, 1)
        c.cx(0, 2)
        c.cx(0, 3)
        c.ry(1)
        c.ry(2)
        c.ry(3)
    c.h(0)
    return c


def run_vqe():
    ham = hamiltonian()
    c = circuit(2)
    vqe = VqeSolver(ham, c)
    vqe.solve()
    vqe.save("vqe")
    print(vqe.res)
    return vqe.circuit


def main():
    # c = run_vqe()
    c = Circuit.load("vqe.circ")
    c.print()


if __name__ == "__main__":
    main()
