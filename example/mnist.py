import struct

import pyopencl as cl
import numpy as np
from pyopencl import cltypes

from nncl import nn, losses, initializer
from nncl.callbacks import PlotCallback
from nncl.layers import Dense, Softmax
from nncl.optimizers import SGD, Anneal
from nncl.util import to_one_hot, get_device


def load_mnist(fname: str) -> np.ndarray:
    with open(fname, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


if __name__ == "__main__":
    device = get_device()
    # device= cl.get_platforms()[0].get_devices()[0]
    np.set_printoptions(linewidth=128)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # Images are 28x28, flatten them to 784
    x_train = load_mnist('../data/mnist/train-images-idx3-ubyte').astype(cltypes.float) / 255.
    x_train.shape = (x_train.shape[0], 784)
    y_train = to_one_hot(load_mnist('../data/mnist/train-labels-idx1-ubyte'))

    x_test = load_mnist('../data/mnist/t10k-images-idx3-ubyte').astype(cltypes.float) / 255.
    x_test.shape = (x_test.shape[0], 784)
    y_test = to_one_hot(load_mnist('../data/mnist/t10k-labels-idx1-ubyte'))

    batch_size = 60
    np.random.seed(0)
    net = nn.Network(ctx=ctx, input_size=784, batch_size=batch_size)
    dense_1 = Dense(ctx, queue,
                    units=128,
                    activation='relu',
                    batch_size=batch_size)
    # dense_2 = Dense(ctx, queue, units=512, activation='relu', batch_size=batch_size)
    # softmax_3 = Dense(ctx, queue, units=10, batch_size=batch_size, activation='hard_sigmoid')
    softmax_3 = Softmax(ctx, queue, units=10, batch_size=batch_size)
    net.add(dense_1)
    # net.add(dense_2)
    net.add(softmax_3)
    net.build()
    net.summary()
    loss = losses.CategoricalCrossentropy(ctx)
    net.train(epochs=100,
              loss=loss,
              optimizer=SGD(),
              batch_size=batch_size,
              x_train=x_train,
              y_train=y_train,
              x_test=x_test,
              y_test=y_test,
              validation_pct=.1,
              validation_method='cross-validation',
              shuffle=True,
              callbacks=[
                  PlotCallback(['batch'], 'Batch Loss', 'Batch No.', 'Loss', True),
                  # PlotCallback(['validation', 'testing'], 'Val/Test Loss', False)
              ])

    input()