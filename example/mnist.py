import struct

import pyopencl as cl
import numpy as np
from pyopencl import cltypes

from nncl import nn, losses, layer, initializer, activation


def load_mnist(fname: str) -> np.ndarray:
    with open(fname, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def to_one_hot(arr: np.ndarray) -> np.ndarray:
    b = np.max(arr) + 1
    return np.eye(b)[arr]


if __name__ == "__main__":
    ctx = cl.Context([cl.get_platforms()[1].get_devices()[0]])
    queue = cl.CommandQueue(ctx)
    net = nn.Network(ctx)

    x_train = load_mnist('../data/mnist/train-images-idx3-ubyte').astype(cltypes.float) / 255.
    x_train.shape = (x_train.shape[0], 784)
    y_train = to_one_hot(load_mnist('../data/mnist/train-labels-idx1-ubyte'))

    x_test = load_mnist('../data/mnist/t10k-images-idx3-ubyte').astype(cltypes.float) / 255.
    x_test.shape = (x_test.shape[0], 784)
    y_test = to_one_hot(load_mnist('../data/mnist/t10k-labels-idx1-ubyte'))

    dense_1 = layer.Dense(ctx, queue, input_width=784, output_width=800, activation='sigmoid')
    dense_2 = layer.Dense(ctx, queue, input_width=800, output_width=10, activation='sigmoid')
    net.add(dense_1)
    net.add(dense_2)
    net.train(epochs=1,
              loss=losses.CategoricalCrossentropy(ctx),
              optimizer=None,
              x_train=x_train,
              y_train=y_train,
              x_test=x_test,
              y_test=y_test)
