from pyopencl.reduction import ReductionKernel
from pyopencl import cltypes
import numpy as np


class Loss:
    def __init__(self, ctx, use_cpu=True):
        self.ctx = ctx
        self.make_reduction_krnl()
        self.use_cpu = use_cpu

    def cpu(self, *args, **kwargs):
        raise NotImplementedError("Please use a subclass")

    def __call__(self, *args, **kwargs):
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

    def cpu(self, y_true, y_pred, idx):
        preds = y_pred.get()
        fields, batch_size = preds.shape
        y_true_np = y_true[idx * batch_size:idx * batch_size + batch_size].get().T
        # print("y_pred:")
        # print(preds)
        # print("y_true:")
        # print(y_true_np)
        mse = np.power(preds - y_true_np, 2) / fields
        # print("MSE:")
        # print(mse)
        # print("Batch Mean Loss:")
        # print(mse.mean())
        return mse.mean()

    def __call__(self, y_true, y_pred, idx):
        if self.use_cpu:
            return self.cpu(y_true, y_pred, idx)
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
            map_expr="y_true[i] * (-log(y_pred[i]))",
            arguments="__global const float* y_true, __global const float* y_pred",
            name="categorical_crossentropy_reduction_kernel"
        )

    def cpu(self, y_true, y_pred, idx):
        preds = y_pred.get()
        batch_size, fields = preds.shape
        y_true_np = y_true[idx * batch_size:idx * batch_size + batch_size].get()
        return (-((y_true_np * np.log(preds)).sum(axis=1))).mean()

    def __call__(self, y_true, y_pred, idx):
        return self.krnl(y_true[idx], y_pred).get()
