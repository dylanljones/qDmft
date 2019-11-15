# -*- coding: utf-8 -*-
"""
Created on 26 Sep 2019
author: Dylan Jones

project: qsim
version: 0.1
"""
import re, os
import numpy as np
import scipy.linalg as la
from scipy import sparse
from qsim.core import *
from qsim import Circuit, Gate, prepare_ground_state, test_vqe
from qsim.fermions import FBasis, Operator, HamiltonOperator
from qsim.twosite_siam import twosite_basis, twosite_hamop, twosite_hamop_sigma

si, sx, sy, sz = pauli


def str_to_list(s, dtype=int):
    """ Convert a string of numbers into list of given data-type

    Parameters
    ----------
    s: str
        String of list
    dtype: type
        Data type of the list

    Returns
    -------
    data_list: list
    """
    if s.strip() == "None":
        return None
    pattern = '-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?'
    return [dtype(x) for x in re.findall(pattern, s)]


def main():
    string = "[-3.614e-08, 6.283185260406262, 2.6779442784401732, -0.0017198866433262613]"
    print(str_to_list(string, float))


if __name__ == "__main__":
    main()
