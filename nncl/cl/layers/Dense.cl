/**
 * Densely connected layer
**/

__kernel void dense_layer_forward(
   __global float* input,   // input buffer
   __global float* weights, // layer weights
//   __global float* biases,  // layer bias
   __global float* output,  // output buffer
    const int input_width   // input width
) {
    // Calculates the activation of the cell
    // Activation is the weighted sum of the inputs
    const int gid = get_global_id(0);
//    const int batch_id = get_global_id(1);

    const int output_width = get_global_size(0);
//    const int batch_size = get_global_size(1);
    float sum = 0;// biases[gid];
    for(int i = 0; i < input_width; i++) {
        sum += weights[gid * input_width + i] * input[i];
    }
//    output[gid * batch_size + batch_id] = ${activation}(sum);
    output[gid] = ${activation}(sum);
}
__kernel void dense_layer_backward(
    __global float* input, // input is the errors
    __global float* weights, // current layers weights to be updated
    __global float* output, // the gradients of this layer
    const int input_width
) {
    const int gid = get_global_id(0);
    const float err = ${derivative}(input[gid]);

}

