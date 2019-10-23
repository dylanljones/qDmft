# -*- coding: utf-8 -*-
"""
Created on 11 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np


class ParameterMap:

    INSTANCE = None

    def __init__(self):
        self.indices = list()
        self.params = list()

    @classmethod
    def instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = cls()
        return cls.INSTANCE

    @property
    def n(self):
        return len(self.indices)

    @property
    def num_params(self):
        return len(self.params)

    @property
    def args(self):
        return [self.get(i) for i in range(self.n)]

    def init(self, *args):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, int):
                args = np.zeros(args)
        self.set(args)

    def set(self, args):
        self.params = list(args)

    def add_param(self, value):
        self.params.append(value)

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, key, value):
        self.params[key] = value

    def add(self, value=None, idx=None):
        next_idx = None
        if idx is None:
            if value is not None:
                next_idx = len(self.params)
                self.params.append(value)
        else:
            next_idx = idx
        self.indices.append(next_idx)

    def get(self, i):
        idx = self.indices[i]
        if idx is None:
            return None
        return self.params[idx]

    def __str__(self):
        return f"Params: {self.params}, Indices: {self.indices}"


pmap = ParameterMap()
