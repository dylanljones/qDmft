# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
import numpy.linalg as la
from scitools import Plot
from dmft.twosite import *


def dmft_step(siam, z, m2):
    gf = impurity_gf_ref(z, siam.u, siam.v)
    gf0 = siam.impurity_gf_free(z)
    sigma = self_energy(gf0, gf)
    qp_weight = quasiparticle_weight(z.real, sigma)
    if qp_weight == 0:
        return siam.v
    else:
        return new_hybridization(qp_weight, m2, siam.v)


def dmft_loop(siam, z, m2, vtol=1e-5):
    v = siam.v
    while True:
        siam.update_hybridization(v)
        v_new = dmft_step(siam, z, m2)
        delta_v = abs(v - v_new)
        v = v_new
        if delta_v <= vtol:
            break
        print(f"v={v:.2f}, Err={delta_v:.2}")
    print("->", v)
    siam.update_hybridization(v)



def main():
    u, t = 4, 1
    tmax, nt = 6, 24
    omax = 6
    omegas = np.linspace(-omax, omax, 10000)
    z = omegas + 0.01j


    siam = TwoSiteSiam.half_filling(u=u, v=t)
    m2 = m2_weight(t)

    dmft_loop(siam, z, m2)

    gf = impurity_gf_ref(z, siam.u, siam.v)
    gf0 = siam.impurity_gf_free(z)
    sigma = self_energy(gf0, gf)

    plot = Plot(xlabel=r"$\omega$")
    plot.grid()
    plot.plot(z.real, -sigma.imag, color="k", label=r"-Im $\Sigma_{imp}(z)$")
    plot.plot(z.real, -gf0.imag, label=r"-Im $G_{imp}^{0}(z)$")
    plot.plot(z.real, -gf.imag, label=r"-Im $G_{imp}(z)$")
    plot.set_limits(0)
    plot.legend()
    plot.show()







if __name__ == "__main__":
    main()
