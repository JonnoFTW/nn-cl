/**
 * Softmax activated dense layer
**/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void layer_forward(
   __global const ${dtype}* input,   // input buffer
   __global const ${dtype}* weights, // layer weights
   __global ${dtype}* biases,  // layer bias
   __global ${dtype}* output,  // output buffer
    const int input_width  // input width
) {

    // calculate product + layer bias
    const int gid = get_global_id(1); // id of the neuron we're evaluating
    const int output_width = get_global_size(1);
    const int batch_id = get_global_id(0);     //  the input sample of the current batch
    const int batch_size = get_global_size(0); // the size of a batch
    // calculate the weighted input for cell gid,
    ${dtype} sum = biases[gid];
    const int output_idx = batch_id *  output_width + gid;
    for(int i = 0; i < input_width; i++) {
        const ${dtype} weight = weights[gid * input_width + i];
        const int input_idx =  (batch_id * input_width) + i;
        const ${dtype} input_val = input[input_idx];
        sum += weight * input_val;
    }
    sum = exp(sum);
    output[output_idx] = sum;

    barrier(CLK_GLOBAL_MEM_FENCE);
    // output array now has exponentiated output
    ${dtype} sumExp = 0;
    for(int i = 0; i < output_width; i++) {
        sumExp += output[batch_id * output_width + i];
    }
    output[output_idx] = sum / sumExp;
}

${dtype} derivative_softmax(${dtype} x) {
    return x;
}