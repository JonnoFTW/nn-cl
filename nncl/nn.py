import numpy as np
import pyopencl as cl
from pyopencl import array, cltypes
from tqdm import tqdm
from typing import List

from nncl.layers.layer import Layer
from nncl.losses import Loss
from nncl.optimizers import Optimizer

mf = cl.mem_flags


class Network:
    def __init__(self, input_size, ctx=None, batch_size=64):
        self.input_size = input_size
        if ctx is None:
            self.ctx = cl.Context([cl.get_platforms()[0].get_devices()[1]])
        else:
            self.ctx = ctx
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.queue = cl.CommandQueue(ctx)
        self.layers: List[Layer] = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build(self):
        # go through the layers and set input/output dimensions appropriately
        if len(self.layers) == 0:
            raise ValueError("You need to add layers before you can build the model")
        units = self.input_size
        for l in self.layers:
            units = l.init(units)

    def summary(self):
        for idx, l in enumerate(self.layers):
            print(f"Layer {idx} {l.name}")
            print(f"\tInputs: {l.input_width}")
            print(f"\tUnits: {l.units}")
            print(f"\tActivation: {l.activation}")

    def forward(self, buf, idx):
        # put x in the buffer
        size = self.layers[0].input_width
        # can probably do better here
        # this only works on pocl because they didn't implement CL_MISALIGNED_SUB_BUFFER_OFFSET 
        #  buf = x.get_sub_region(size * idx, size)
        offset = cltypes.int(self.batch_size * size * idx)
        for idx, l in enumerate(self.layers):
            input_np = np.zeros((self.batch_size, l.input_width), dtype=cltypes.float)
            cl.enqueue_copy(self.queue, input_np, buf, device_offset=offset * 4)
            # print(f"Layer {idx}")
            # print(f"Input: cols={l.input_width} inputs rows={l.batch_size} samples/batch\n", input_np)
            buf = l(buf, offset).data
            offset = cltypes.int(0)
            weights = l.get_weights().reshape(l.units, l.input_width)
            bias = l.get_bias()
            output = l.get_output()
            # print(f"\nWeights: (rows={l.units} units, cols={l.input_width} inputs)\n", weights)
            # print("Biases:\n", bias)
            # print(f"Output: (cols={l.batch_size} samples/batch, rows={l.units} units)\n", output)
            # print("Expected:\n", np.clip(weights.dot(input_np) + bias, 0, a_max=None))
            # print()

        # output is the output of the last layer
        return self.layers[-1].output

    def backward(self, err, x_data: cl.Buffer, idx, y_true, optimizer: Optimizer):
        """

        :param err: the loss error
        :param x_data:
        :param idx:
        :param y_true:
        :param optimizer:
        :return:
        """
        size = self.layers[0].input_width
        x_data = x_data.get_sub_region(size * idx, size)
        y_data = y_true[idx].data
        optimizer(self, err, x_data, y_data, idx)

    def train(self, epochs: int,
              loss: Loss,
              optimizer: Optimizer,
              x_train,
              y_train,
              x_test,
              y_test,
              batch_size: int = 1,
              shuffle: bool=False):
        """

        :param epochs: number of epochs to run
        :param loss:  a loss function
        :param optimizer: the optimizer to use
        :param x_train: a 2D array of shape (rows, features)
        :param y_train: a 2d array of shape (rows, output features),
                output_features is the number of values we want to predict
        :param x_test: testing data inputs
        :param y_test: testing data true values
        :return: None

        For example, our input might be:
        x_train = [
            [0,1,1],
            [0,2,1],
            [1,2,1],
            [0,3,4],
        ]
        That is 4 rows with 3 features each, we might do a binary classification on this:
        y_train = [
            [0,1],
            [0,1],
            [1,0],
            [0,1]
        ]
        That is, each training input maps to one of these
        All this will be copied to the device

        """
        x_train = x_train.astype(cltypes.float)
        y_train = y_train.astype(cltypes.float)
        x_test = x_test.astype(cltypes.float)
        y_test = y_test.astype(cltypes.float)

        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            raise ValueError("X and Y for test/train must be same length")
        train_rows = x_train.shape[0]
        if train_rows % batch_size != 0:
            raise ValueError("Training dataset must have rows divisible by batch size")
        input_features = cltypes.uint(x_train.shape[1])
        output_features = cltypes.uint(y_train.shape[1])
        if input_features != self.layers[0].input_width:
            raise ValueError(
                f"Input features (provided={input_features}) must be the same as layer_0 input width (required={self.layers[0].input_width})")
        # Just copy all training and all testing data to the device
        x_train_gpu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_train)
        # y_train_gpu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_train)
        # y_train_gpu = array.Array(self.queue, y_train.shape, dtype=cltypes.float, data=y_train)
        # x_train_gpu = array.Array(self.queue, data=y_train_gpu_buf, shape=y_train.shape, dtype=cltypes.float)

        # should probably check that our data won't exceed available device memory,
        # transparently queue up more data once it's been used
        # get ~1685 row/s on pocl, intel i7-4770
        y_train_gpu = array.to_device(self.queue, y_train)
        # y_train_gpu_arr = array.Array(self.queue, y_train.size, cltypes.float, data=y_train_gpu)
        for i in tqdm(range(epochs), desc='Epoch: ', position=0):
            # shuffle
            if shuffle:
                # shuffle samples within x_train_gpu and the corresponding y_train_gpu
                pass
            for idx in tqdm(range(train_rows // batch_size), desc='Batch: ', position=1, unit=' batch'):
                idx = cltypes.uint(idx)  # idx here is the batch number, the nth batch
                # copy all of these to the device?
                output = self.forward(x_train_gpu, idx)
                err = loss.cpu(y_train_gpu, output, idx=idx)
                optimizer(self, err, x_train_gpu, y_train_gpu, idx)
                # self.backward(err, x_data=x_train_gpu, idx=idx, y_true=y_train_gpu, optimizer=optimizer)
                # print(err)
            # print(err)
