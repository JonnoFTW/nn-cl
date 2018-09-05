import pyopencl as cl
import numpy as np
from pyopencl import cltypes
import os
from nncl import nn, losses
from nncl.layers import layer

if __name__ == "__main__":
    ctx = cl.Context([cl.get_platforms()[1].get_devices()[0]])
    queue = cl.CommandQueue(ctx)
    net = nn.Network(ctx)
    iris = np.loadtxt(os.path.dirname(os.path.realpath(__file__))+'/../data/iris.csv', skiprows=1, delimiter=',', dtype=cltypes.float)
    np.random.seed(420)
    np.random.shuffle(iris, )
    x = iris[:, :-1]
    y = iris[:, -1:]
    # convert y to sparse categorical,
    # ie. row with class 1 will have [0,1,0]
    # 3 output classes
    # sparse_y = np.zeros((x.shape[0], 3))
    # for idx, c in enumerate(iris[:, -1]):
    #     sparse_y[idx, int(c)] = 1
    split_idx = int(0.33 * x.shape[0])
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    batch_size = 5
    # 4 features, 1 class
    dense_1 = layer.Dense(ctx, queue, input_width=4, output_width=16, activation='hard_sigmoid', batch_size=batch_size)
    soft_2 = layer.Softmax(ctx, queue, input_width=16, output_width=3, batch_size=batch_size)
    net.add(dense_1)
    net.add(soft_2)
    net.train(epochs=1,
              loss=losses.CategoricalCrossentropy(ctx),
              optimizer=None,
              x_train=x_train,
              batch_size=batch_size,
              y_train=y_train,
              x_test=x_test,
              y_test=y_test)
