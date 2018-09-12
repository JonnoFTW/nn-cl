import numpy as np
import pyopencl as cl
from pyopencl import array
from datetime import datetime

device = cl.get_platforms()[1].get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
print(device)

def main(dtype, s):
    shape = (s, s)
    arr_np_a = np.random.random(shape).astype(dtype=dtype)
    arr_np_b = np.random.random(shape).astype(dtype=dtype)
    arr_g_a = array.to_device(queue, arr_np_a)
    arr_g_b = array.to_device(queue, arr_np_b)
    start = datetime.now()
    array.dot(arr_g_a, arr_g_b)
    diff = datetime.now() - start

    return diff


char = np.int8
uchar = np.uint8
short = np.int16
ushort = np.uint16
int = np.int32
uint = np.uint32
long = np.int64
ulong = np.uint64
half = np.float16
float = np.float32
double = np.float64

if __name__ == "__main__":
    s = 1024

    times = [(main(i, s), i) for i in [char, uchar, short, ushort, int, uint, long, ulong, float, double]]
    for i in times:
        print(i)
    for time, dtype in sorted(times):
        print(dtype, "took", time)
