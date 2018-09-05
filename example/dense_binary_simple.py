import struct

import pyopencl as cl
import numpy as np
from pyopencl import cltypes

from nncl import nn, losses, initializer
from nncl.layers import Dense, Softmax
from nncl.optimizers import SGD
from nncl.util import get_device
if __name__ == "__main__":
    device = get_device()

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # x_train = np.array([[0, 0],
    #                     [0, 1],
    #                     [1, 1],
    #                     [1, 0]]).astype(cltypes.float)
    # y_train = to_one_hot(np.array([0, 1, 0, 1])).astype(cltypes.float)

    x_train = np.random.uniform(0, 1, (16, 2))
    x_train[0] = [1]
    x_train[1] = [2]
    y_train = (x_train[:,0] > 0.5).reshape(16,1)
    batch_size = 4
    np.random.seed(0)
    net = nn.Network(ctx=ctx, input_size=2, batch_size=batch_size)
    dense_1 = Dense(ctx, queue,
                    units=6,
                    activation='relu',
                    batch_size=batch_size)
    sigmoid_2 = Dense(ctx, queue, units=1, batch_size=batch_size, activation='sigmoid')
    net.add(dense_1)
    net.add(sigmoid_2)
    net.build()
    # net.summary()
    net.train(epochs=1,
              loss=losses.MSE(ctx),
              optimizer=SGD(),
              batch_size=batch_size,
              x_train=x_train,
              y_train=y_train,
              x_test=x_train,
              y_test=y_train)
