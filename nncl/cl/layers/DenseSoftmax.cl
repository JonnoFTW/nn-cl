/**
 * Softmax activated dense layer
**/

__kernel void softmax_layer_forward(
   __global float* input,   // input buffer
   __global float* weights, // layer weights
//   __global float* biases,  // layer bias
   __global float* output,  // output buffer
    const int input_width   // input width
) {

    // calculate product + layer bias
    const int gid = get_global_id(0);
    const int output_width = get_global_size(0);
    // calculate the output for cell gid,
    float sum = 0; // biases[gid];
    for(int i = 0; i < input_width; i++) {
        sum += weights[gid * input_width + i] * input[i];
    }

    output[gid] = exp(sum);
    barrier(CLK_GLOBAL_MEM_FENCE);
    // calc the sum of outputs
    float output_total = 0;
    for(int i = 0; i < output_width; i++) {
        output_total += output[i];
    }
    output[gid] /= output_total;
}
__kernel void softmax_layer_backward() {

}