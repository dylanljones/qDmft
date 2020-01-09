# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
import scipy.linalg as la
from scipy import optimize
from qsim.core.circuit import Circuit


class VqeResult(optimize.OptimizeResult):

    def __init__(self, sol, exact=0):
        super().__init__(x=sol.x, success=sol.success, message=sol.message,
                         nfev=sol.nfev, nit=sol.nit)
        self.exact = exact
        self.value = sol.fun

    @property
    def error(self):
        return abs(self.value - self.exact)

    @property
    def rel_error(self):
        return abs((self.value - self.exact) / self.exact)

    def string(self, d=5, maxvals=10):
        popt_str = ", ".join([f"{x:.{d}}" for x in self.x[:maxvals]])
        if len(self.x) > maxvals:
            popt_str += ", ..."
        lines = list()
        lines.append(f"Success: {self.success} (nfev={self.nfev}, nit={self.nit})")
        lines.append(f"Message: {self.message}")
        lines.append(f"x:       {popt_str}")
        lines.append(f"Exact:   {self.exact:.{d}}")
        lines.append(f"Result:  {self.value}")
        lines.append(f"Error:   {self.error:.{d}} ({100 * self.rel_error:.2f}%)")

        line = "-" * (max([len(x) for x in lines]) + 1)
        string = "\n".join(lines)
        return f"{line}\n{string}\n{line}"

    def __str__(self):
        return self.string()


class VqeSolver:

    def __init__(self, ham, num_clbits=None):
        self.ham = None
        self.exact = 0
        self.circuit = None
        self.sol = None

        self.setup(ham, num_clbits)

    def setup(self, ham, num_clbits=None):
        # Calculate exact groundstate
        eigvals, eigstates = np.linalg.eig(ham)
        gs = np.min(eigvals).real

        # Setup vqe-circuit
        num_qubits = int(np.log2(ham.shape[0]))
        circuit = Circuit(num_qubits, num_clbits)

        self.ham = ham
        self.exact = gs
        self.circuit = circuit
        self.sol = None

    @property
    def n_params(self):
        return self.circuit.n_params

    @property
    def success(self):
        return self.sol is not None and self.sol.success

    @property
    def atol(self):
        return abs(self.exact - self.circuit.expectation(self.ham))

    @property
    def x(self):
        return self.sol.x

    def __str__(self):
        string = "VQE-Solver:"
        if self.sol is not None:
            string += "\n" + str(self.sol)
        else:
            string += " Ready!"
        return string

    def set_circuit(self, circuit):
        self.circuit = circuit

    def expectation(self, params):
        self.circuit.set_params(params)
        self.circuit.run_circuit()
        return self.circuit.expectation(self.ham)

    def minimize(self, x0=None, **kwargs):
        if x0 is None:
            x0 = np.random.uniform(0, np.pi, size=self.n_params)
        sol = optimize.minimize(self.expectation, x0=x0, **kwargs)
        self.sol = VqeResult(sol, self.exact)
        return self.sol

    def check(self, atol=1e-2):
        return self.atol <= atol

    def save_circuit(self, name):
        return self.circuit.save(name + ".circ")

    def save_state(self, name):
        self.circuit.save_state(name)
