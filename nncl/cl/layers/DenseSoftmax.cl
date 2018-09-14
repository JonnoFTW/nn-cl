/**
 * Softmax activated dense layer
**/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void softmax_layer_forward(
   __global const ${dtype}* input,   // input buffer
   __global const ${dtype}* weights, // layer weights
   __global ${dtype}* biases,  // layer bias
   __global ${dtype}* output,  // output buffer
    const int input_width,  // input width
    const int offset // the offset into the input data
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
        const int input_idx =  offset + (batch_id * input_width) + i;
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
__kernel void softmax_layer_backward(
    __global const ${dtype}* y_pred, // the probabilities, `output` in the forward step
    __global const ${dtype}* x_train,
    __global const ${dtype}* y_true,
    __global ${dtype}* weights,
    __global ${dtype}* deltas,
    const int x_width,
    const ${dtype} lr, // the learning rate
    const ${dtype} reg // the regularization value
) {
    // Diff is p_x-1 if y_true matches
    const int gid = get_global_id(0); // the output cell idx
    const int output_width = get_global_size(0);
    // for each W of this cell
 //   for(int i = 0; i < o)
    ${dtype} diff = y_pred[gid] - 1*(y_true[gid] == 1.0);
    ${dtype} dW = reg; // this needs to be over the transpose: dW = X.T * W
    for(int i = 0; i < x_width; i++) {
        dW += diff * x_train[i];
    }
    deltas[gid] = dW;
    weights[gid] -= lr * dW;
}