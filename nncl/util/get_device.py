import os, re


def get_device():
    """
    Get the device from the user environment
    Should be in format:
    opencl0:0
    :return:
    """
    var = os.getenv('device', None)
    if var is None:
        return 0, 0
    else:
        try:
            return [int(i) for i in re.findall(r"opencl(\d+):(\d+)", var)[0]]
        except:
            raise ValueError(
                "OpenCL environment name must be in format opencl:m:n where m is platform number, n is device number")
