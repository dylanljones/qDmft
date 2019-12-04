# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import numpy as np
from scitools import Plot
from dmft.twosite import *
from dmft import TwoSiteSiam


def main():
    u, t = 4, 1
    z = np.linspace(-5, 5, 1000) + 0.01j
    m2 = m2_weight(t)

    siam = TwoSiteSiam.half_filling(u=u, v=t)

    gf = impurity_gf_ref(z, siam.u, siam.v)
    gf0 = siam.impurity_gf_free(z)
    sigma = self_energy(gf0, gf)

    plot = Plot()
    plot.plot(z.real, -gf.imag)
    plot.plot(z.real, -gf0.imag)
    plot.plot(z.real, -sigma.imag, color="k")
    plot.show()


if __name__ == "__main__":
    main()
