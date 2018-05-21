import numpy as np
import pyopencl as cl
from pyopencl import array, cltypes
from tqdm import tqdm

from nncl.layer import Layer
from nncl.losses import Loss

mf = cl.mem_flags


class Network:
    def __init__(self, ctx=None):
        if ctx is None:
            self.ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
        else:
            self.ctx = ctx
        self.queue = cl.CommandQueue(ctx)
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build(self):
        pass

    def forward(self, x):
        # put x in the buffer
        buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        size = x.shape[0]
        for l in self.layers:
            # input_np = np.zeros(size, dtype=cltypes.float)
            # cl.enqueue_copy(self.queue, input_np, buf)
            # print("\nInput:  ", input_np)
            buf = l(buf)
            # weights = l.get_weights().reshape(l.output_width, l.input_width)
            # output = l.get_output()
            # print("Weights:", weights)
            # print("Output: ", output)
            # print("Expected: ", np.tanh(weights.dot(input_np)))
            # print()
            size = l.output_width

        # buf is now a pointer to the cl buffer of the output
        # should probably calculate the loss on the device
        # return the output of the last layer
        return buf, size

    def backward(self, ):
        pass

    def train(self, epochs: int, loss: Loss, optimizer, x_train, y_train, x_test, y_test, batch_size: int = 1):
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
        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            raise ValueError("X and Y for test/train must be same length")
        train_rows = x_train.shape[0]
        if train_rows % batch_size != 0:
            raise ValueError("Training dataset must have rows divisible by batch size")
        input_features = x_train.shape[1]
        output_features = y_train.shape[1]
        if input_features != self.layers[0].input_width:
            raise ValueError("Input features must be the same as layer_0 input width")
        for i in tqdm(range(epochs), desc='Epoch: ', position=0):
            for idx in tqdm(range(train_rows), desc='Row: ', position=1):
                # copy all of these to the device?
                x = x_train[idx]
                y = y_train[idx]
                y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
                output, output_width = self.forward(x)  # this is a buffer
                # output and y need to use opencl.array
                output = array.Array(self.queue, output_width, cltypes.float, data=output)
                y_gpu = array.Array(self.queue, y.shape[0], cltypes.float, data=y_buf)
                err = loss(output, y_gpu, output_width)
                print(err)
