#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void get_gradients(
    __global const ${dtype}* gradients, //gradients input
    __global ${dtype}* weights, // weights to be updated
    ${dtype} lr // the learning rate
) {
    const int gid = get_global_id(0);
    weights[gid] = weights[gid] - (${derivative}(gradients[gid]) * lr);
}

