from pyopencl.reduction import ReductionKernel
from pyopencl import cltypes
import numpy as np


class Loss:
    def __init__(self, ctx):
        self.ctx = ctx
        self.make_reduction_krnl()

    def __call__(self, y_true, y_pred, n):
        raise NotImplementedError("Please use an actual loss")

    def make_reduction_krnl(self):
        raise NotImplementedError("Please use a subclass that implements loss")


class MSE(Loss):
    def make_reduction_krnl(self):
        self.krnl = ReductionKernel(
            self.ctx,
            cltypes.float,
            neutral="0",
            reduce_expr="a+b",
            map_expr="pow(y_pred[i] - y_true[i], 2)",
            arguments="__global const float* y_true, __global const float* y_pred",
            name="mse_reduction_kernel"
        )

    def __call__(self, y_true, y_pred, n):
        return self.krnl(y_true, y_pred).get() / n


class CategoricalCrossentropy(Loss):
    """
    Categorical Cross Entropy:
    - sum(y_true * log2(y_pred))
    """

    def make_reduction_krnl(self):
        self.krnl = ReductionKernel(
            self.ctx,
            cltypes.float,
            neutral="0",
            reduce_expr="a+b",
            # p is the true distribution, q is predicted
            map_expr="y_true[i] * (-log2(y_pred[i]))",
            arguments="__global const float* y_true, __global const float* y_pred",
            name="categorical_crossentropy_reduction_kernel"
        )

    def cpu(self, y_true, y_pred, n):
        return -((y_true * np.log2(y_pred)).sum())

    def __call__(self, y_true, y_pred, n):
        return self.krnl(y_true, y_pred).get()
