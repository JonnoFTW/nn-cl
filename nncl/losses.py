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

    def deltas(self, y_pred, y_true):
        """
        Get deltas
        :param y_pred:
        :param y_true:
        :return:
        """
        raise NotImplementedError("Please use an actual loss function")


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

    def cpu(self, y_true, y_pred):
        y_pred_np = y_pred.get()
        fields = y_pred_np.shape[1]
        y_true_np = y_true.get().T
        # print("y_pred:")
        # print(preds)
        # print("y_true:")
        # print(y_true_np)
        mse = np.power(y_pred_np - y_true_np, 2) / fields
        # print("MSE:")
        # print(mse)
        # print("Batch Mean Loss:")
        # print(mse.mean())
        return mse.mean()

    def __call__(self, y_true, y_pred):
        return self.krnl(y_true, y_pred).get() / y_pred.shape[0]


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

    def errors(self, y_pred, y_true):
        return y_pred - y_true

    def cpu(self, y_true, y_pred):
        y_pred_np = y_pred.get()
        y_true_np = y_true.get()
        out = (-((y_true_np * np.log(y_pred_np)).sum(axis=1)))
        # for o in zip(out, np.argmax(y_pred_np, axis=1), np.argmax(y_true_np, axis=1), y_pred_np):
        #     print(f"{o[3]} Err: {o[0]} expected: {o[2]} predicted: {o[1]}")
        # print("Errors:\n", y_true - y_pred)
        return out.mean()

    def __call__(self, y_true, y_pred, idx):
        return self.krnl(y_true[idx], y_pred).get()
