# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from scitools import Plot
from qsim import kron, pauli
from qsim import Circuit, VqeSolver, prepare_ground_state, test_vqe
from qsim.fermions import HamiltonOperator
VQE_FILE = "circuits/twosite_vqe_2"

si, sx, sy, sz = pauli

X0 = [5.7578, 3.1416, 3.6519, 2.8804, 4.5688, 5.9859, 4.187, 4.7124]

# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def hamiltonian_sig(u=4, eps=2, mu=2, v=1):
    u_op = 1/2 * (kron(si, sz, si, sz, si) - kron(si, sz, si, si, si) - kron(si, si, si, sz, si))
    mu_op = (kron(si, sz, si, si, si) + kron(si, si, si, sz, si))
    eps_op = (kron(si, si, sz, si, si) + kron(si, si, si, si, sz))
    v_op = kron(si, sx, sx, si, si) + kron(si, sy, sy, si, si) \
            + kron(si, sy, sy, sx, sx) + kron(si, si, si, sy, sy)
    return 1/2 * (u * u_op + mu * mu_op - eps * eps_op + v * v_op)


def config_vqe_circuit(vqe):
    c = vqe.circuit
    c.ry([1, 2, 3, 4])
    c.cx(3, 4)
    c.ry([3, 4])
    c.cx(1, 3)
    c.ry([1, 3])
    c.cx(1, 2)
    return c


# =========================================================================
#                            TIME EVOLUTION
# =========================================================================


def run_gf_circuit(v=4, tau=6, n=12):
    arg = v/2 * tau/n
    gf = np.zeros(n)
    for i in range(n):
        x = measure_gf(arg, step=i)
        print(x)
        gf[i] = x.imag
    plot = Plot()
    plot.plot(gf)
    plot.show()



def time_evolution_circuit(c, arg, step=1):
    for i in range(step):
        # print("Step", i)
        c.xy([[1, 2], [3, 4]], [arg, arg])
        c.b([1, 3], arg)
    return c


def measure_circuit(arg, step=1, alpha="x", beta="x", mbasis=sy):
    n = 100
    c = Circuit(5, 1)
    c.backend.load_state("state.npy")
    c.h(0)
    c.add_gate(f"c{alpha}", 1, con=0, trigger=0)
    time_evolution_circuit(c, arg, step)
    c.add_gate(f"c{beta}", 1, con=0, trigger=1)
    c.h(0)
    data = np.zeros(n)
    for i in range(n):
        data[i] = c.backend.measure_qubit(c.qureg[0], mbasis, shadow=True)
    return np.mean(data, axis=0)


def measure_gf_greater(arg, step=1):
    g1 = measure_circuit(arg, step, "x", "x")
    g2 = measure_circuit(arg, step, "y", "x")
    g3 = measure_circuit(arg, step, "x", "y")
    g4 = measure_circuit(arg, step, "y", "y")
    # print("Greater")
    # print("xx", g1)
    # print("yx", g2)
    # print("xy", g3)
    # print("yy", g4)
    return -0.25j * (g1 + 1j*g2 - 1j*g3 + g4)


def measure_gf_lesser(arg, step=1):
    g1 = measure_circuit(arg, step, "x", "x")
    g2 = measure_circuit(arg, step, "x", "y")
    g3 = measure_circuit(arg, step, "y", "x")
    g4 = measure_circuit(arg, step, "y", "y")
    # print("Lesser")
    # print("xx", g1)
    # print("xy", g2)
    # print("yx", g3)
    # print("yy", g4)
    return +0.25j * (g1 - 1j*g2 + 1j*g3 + g4)


def measure_gf(arg, step):
    gf_g = measure_gf_greater(arg, step)
    gf_l = measure_gf_lesser(arg, step)
    return gf_g - gf_l


def main():
    tau = 6
    v = 4
    n = 12
    arg = v/2 * tau/n

    run_gf_circuit(v, tau, n)



if __name__ == "__main__":
    main()
