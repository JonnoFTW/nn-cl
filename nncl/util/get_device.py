import pyopencl as cl
import os, re


def get_device():
    """
    Get the device from the user environment
    Should be in format:
    opencl0:0
    :return: platform and device numbers
    """
    var = os.getenv('device', None)
    if var is None:
        pid, did = 0, 0
    else:
        try:
            pid, did = [int(i) for i in re.findall(r"opencl(\d+):(\d+)", var)[0]]
        except:
            raise ValueError(
                "OpenCL environment name must be in format opencl:m:n where m is platform number, n is device number")
    platform = cl.get_platforms()[pid]
    return platform.get_devices()[did]
