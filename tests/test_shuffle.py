import humanize
import numpy as np
import pyopencl as cl
from pyopencl import cltypes
from pyopencl import array
from datetime import datetime

device = cl.get_platforms()[1].get_devices()[0]
print(device)
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
np.set_printoptions(suppress=True)
with open('../nncl/cl/utils/shuffle.cl') as infile:
    src = infile.read()

shuffle_prog = cl.Program(ctx, src).build()
shuffle_krnl = shuffle_prog.shuffle_data


def read_only_arr(numbytes):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY, numbytes)


def serial_shuffle(x_data, swaps, verbose=False):
    for i, j in zip(swaps[0::2], swaps[1::2]):
        x_data[i], x_data[j] = x_data[j], x_data[i].copy()
    return x_data


def shuffle(x_data, rows, cols):
    """
    Odd sized row count will not have 1 row shuffled
    :param x_data:
    :param rows:
    :param cols:
    :param swaps_g:
    :return:
    """
    swaps_np = np.arange(rows, dtype=cltypes.uint)
    np.random.shuffle(swaps_np)
    swaps_g = array.to_device(queue, swaps_np, allocator=read_only_arr)
    e1 = shuffle_krnl(queue, (cols, len(swaps_np) // 2), None, x_data, swaps_g.data)
    e1.wait()
    return swaps_g


def test_shuffle(rows, cols, verbose=False):
    x_data_np = np.arange(rows * cols, dtype=cl.cltypes.float).reshape(rows, cols)
    x_data_np_orig = x_data_np.copy()
    x_data = array.to_device(queue, x_data_np)
    if verbose:
        print("Before:")
        print("X:")
        for idx, row in enumerate(x_data_np):
            print(idx, row)
    print(f"Rows: {rows}, cols={cols} {humanize.naturalsize(x_data_np.nbytes)}")
    start = datetime.now()
    swaps = shuffle(x_data.data, rows, cols)
    print("\tCL Took:", datetime.now() - start)
    swap_np = swaps.get()
    if verbose:
        print("After:")
        for idx, row in enumerate(x_data.get()):
            print(idx, row)
        print("swaps:")
        print(swap_np)
    start = datetime.now()
    np.random.shuffle(x_data_np)
    print("\tNP Took: ", datetime.now() - start)
    start = datetime.now()
    s_shuf = serial_shuffle(x_data_np_orig, swap_np, verbose)
    print("\tSerial took: ", datetime.now() - start)
    if verbose:
        print("Serial")
        for idx, row in enumerate(s_shuf):
            print(idx, row)
    assert np.equal(x_data.get(), s_shuf).all()


if __name__ == "__main__":
    test_shuffle(11, 1, True)
    test_shuffle(10, 3, True)
    test_shuffle(10000, 1000)
    test_shuffle(50000, 1200)
    test_shuffle(1200, 50000)
    test_shuffle(784, 60000)
    test_shuffle(60000, 784)
#
