# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import os
import numpy as np
from qsim import *
from scitools import Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_DATA_PATH = "data\\figs\\"
IMG_PATH = r"D:\Dropbox\Uni\Master\Master Thesis\Latex Script\img"


def plot_bell_measurement(new_data=True, n_values=(1, 100, 1000)):
    if new_data:
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.mz()
        for n in n_values:
            res = c.run(n)
            res.save(IMG_DATA_PATH + f"bell_meas_{n}.npy")
    plot = Plot()
    plot.latex_plot()
    for n in n_values:
        res = Result.laod(IMG_DATA_PATH + f"bell_meas_{n}.npy")
        plot = res.show_histogram(False)
        plot.latex_plot(0.5, ratio=1, font="lmodern", fontsize=11)
        plot.tight()
        plot.save(os.path.join(IMG_PATH, f"bell_meas_{n}.png"))


def plot_density_matrix(res, width=0.8, show_zero=True, elev=30, azim=20):
    rho = res.density_matrix()

    n = rho.shape[0]
    xx, yy = np.meshgrid(range(n), range(n))
    if not show_zero:
        xx = xx[rho != 0]
        yy = yy[rho != 0]
        rho = rho[rho != 0]

    ticks = np.arange(n) + 0.5 * width
    lims = - 0.25 * width, (n - 1) + 1.25 * width
    labels = res.labels

    plot = Plot(proj="3d")
    plot.bar3d(xx, yy, rho, width, width, color="midnightblue")
    plot.set_limits(lims, lims)
    plot.set_ticks(ticks, ticks)
    plot.set_ticklabels(labels, labels)
    # plot.set_view(elev, azim)
    plot.show()


def main():
    # plot_bell_measurement()
    c = Circuit(2, 2)
    c.h(0)
    c.cx(0, 1)
    c.mz()
    res = c.run(1)
    plot_density_matrix(res)


if __name__ == "__main__":
    main()
