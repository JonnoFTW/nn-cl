from nncl import optimizers
import pyopencl as cl
import numpy as np
from pyopencl import cltypes, array

device = cl.get_platforms()[1].get_devices()[0]
ctx = cl.Context([device])

queue = cl.CommandQueue(ctx)


def read_only_arr(numbytes):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY, numbytes)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    anneal = optimizers.Anneal(queue, None)

    # make an array of floats
    a = np.arange(0, 32, dtype=cltypes.float)
    a_gpu = array.to_device(queue, a, allocator=read_only_arr)
    out_gpu = array.empty_like(a_gpu, queue)
    print(a)
    anneal.add_krnl(queue, (a.size,), None, a_gpu.data, out_gpu.data, cltypes.float(np.random.randn())).wait()
    print(out_gpu.get())
