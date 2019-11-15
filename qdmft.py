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

VQE_FILE = "circuits/twosite_vqe"

si, sx, sy, sz = pauli

X0 = [5.7578, 3.1416, 3.6519, 2.8804, 4.5688, 5.9859, 4.187, 4.7124]

# =========================================================================
#                         GROUND STATE PREPARATION
# =========================================================================


def twosite_hamop(basis):
    (c1u, c2u), (c1d, c2d) = basis.annihilation_ops()
    u_op = (c1u.dag * c1u * c1d.dag * c1d)
    mu_op = (c1u.dag * c1u) + (c1d.dag * c1d)
    eps_op = (c2u.dag * c2u) + (c2u.dag * c2u)
    v_op = (c1u.dag * c2u) + (c2u.dag * c1u) + (c1d.dag * c2d) + (c2d.dag * c1d)
    return HamiltonOperator(u=u_op, mu=-mu_op, eps=eps_op, v=v_op)


def hamiltonian_sig(u=4, eps=2, mu=2, v=1):
    u_op = 1/2 * (kron(sz, si, sz, si) - kron(sz, si, si, si) - kron(si, si, sz, si))
    mu_op = (kron(sz, si, si, si) + kron(si, si, sz, si))
    eps_op = (kron(si, sz, si, si) + kron(si, si, si, sz))
    v_op = kron(sx, sx, si, si) + kron(sy, sy, si, si) + kron(sy, sy, sx, sx) + kron(si, si, sy, sy)
    return 1/2 * (u * u_op + mu * mu_op - eps * eps_op + v * v_op)


def config_vqe_circuit(vqe):
    c = vqe.circuit
    c.ry([0, 1, 2, 3])
    c.cx(2, 3)
    c.ry([2, 3])
    c.cx(0, 2)
    c.ry([0, 2])
    c.cx(0, 1)
    return c


# =========================================================================
#                            TIME EVOLUTION
# =========================================================================


def time_evolution_circuit(arg, step):
    c = Circuit(5, 1)
    c.h(0)
    c.cx(0, 1)
    for i in range(step):
        c.xy([[1, 2], [3, 4]], [arg, arg])
        c.b([1, 3], arg)
    c.cy(0, 1)
    c.h(0)
    return c


def get_twosite_circuit(arg, step):
    c = prepare_ground_state()
    c.add_qubit(0)
    c.append(time_evolution_circuit(arg, step))
    return c


def main():
    tau = 6
    v = 4
    n = 20
    arg = v/2 * tau/n
    ham = hamiltonian_sig()
    c = prepare_ground_state(ham, config_vqe_circuit, x0=X0, file=VQE_FILE, new=True, clbits=0)
    c.add_qubit(0, add_clbit=True)

    u = time_evolution_circuit(np.pi/2, 2)
    c.append(u)
    c.print(show_args=False)


    # s.apply_gate(xy_gate(np.pi/3))


if __name__ == "__main__":
    main()
