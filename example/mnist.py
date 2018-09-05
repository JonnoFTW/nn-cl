import struct

import pyopencl as cl
import numpy as np
from pyopencl import cltypes

from nncl import nn, losses, initializer
from nncl.layers import Dense, Softmax
from nncl.util import to_one_hot


def load_mnist(fname: str) -> np.ndarray:
    with open(fname, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


if __name__ == "__main__":
    platform = cl.get_platforms()[0]
    indent = ''
    print(indent + '{} ({})'.format(platform.name, platform.vendor))
    indent = '\t'
    print(indent + 'Version: ' + platform.version)
    print(indent + 'Profile: ' + platform.profile)
    print(indent + 'Extensions: ' + ', '.join(platform.extensions.strip().split(' ')))
    for device in platform.get_devices():
        # device = platform.get_devices()[0]
        print(indent + '{} ({})'.format(device.name, device.vendor))

        indent = '\t\t\t'
        flags = [('Version', device.version),
                 ('Type', cl.device_type.to_string(device.type)),
                 ('Extensions', ', '.join(device.extensions.strip().split(' '))),
                 ('Memory (global)', str(device.global_mem_size)),
                 ('Memory (local)', str(device.local_mem_size)),
                 ('Address bits', str(device.address_bits)),
                 ('Max work item dims', str(device.max_work_item_dimensions)),
                 ('Max work group size', str(device.max_work_group_size)),
                 ('Max compute units', str(device.max_compute_units)),
                 ('Driver version', device.driver_version),
                 ('Image support', str(bool(device.image_support))),
                 ('Little endian', str(bool(device.endian_little))),
                 ('Device available', str(bool(device.available))),
                 ('Compiler available', str(bool(device.compiler_available)))]

        [print(indent + '{0:<25}{1:<10}'.format(name + ':', flag)) for name, flag in flags]
        # print("Device: ", device)
        # device.
        # print("  MEM BASE ADDR ALIGN", device.get_info(cl.device_info.MEM_BASE_ADDR_ALIGN))
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        # Images are 28x28, flatten them to 784
        x_train = load_mnist('../data/mnist/train-images-idx3-ubyte').astype(cltypes.float) / 255.
        x_train.shape = (x_train.shape[0], 784)
        y_train = to_one_hot(load_mnist('../data/mnist/train-labels-idx1-ubyte'))

        x_test = load_mnist('../data/mnist/t10k-images-idx3-ubyte').astype(cltypes.float) / 255.
        x_test.shape = (x_test.shape[0], 784)
        y_test = to_one_hot(load_mnist('../data/mnist/t10k-labels-idx1-ubyte'))

        batch_size = 2
        np.random.seed(0)
        net = nn.Network(ctx=ctx, input_size=784, batch_size=batch_size)
        dense_1 = Dense(ctx, queue,
                        units=16,
                        activation='relu',
                        batch_size=batch_size)
        # dense_2 = Dense(ctx, queue, units=512, activation='relu', batch_size=batch_size)
        softmax_3 = Softmax(ctx, queue, units=10, batch_size=batch_size)
        net.add(dense_1)
        # net.add(dense_2)
        net.add(softmax_3)
        net.build()
        net.summary()
        net.train(epochs=1,
                  loss=losses.CategoricalCrossentropy(ctx),
                  optimizer=None,
                  batch_size=batch_size,
                  x_train=x_train,
                  y_train=y_train,
                  x_test=x_test,
                  y_test=y_test)
        print()
