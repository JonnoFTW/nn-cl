/**
 * Softmax activated dense layer
**/

__kernel void softmax_layer_forward(
   __global const float* input,   // input buffer
   __global const float* weights, // layer weights
//   __global float* biases,  // layer bias
   __global float* output,  // output buffer
    const int input_width,  // input width
    const int offset // the offset into the input data
//     __local float* exponents
) {

    // calculate product + layer bias
    const int gid = get_global_id(0); // id of the neuron we're evaluating
    const int output_width = get_global_size(0);
    const int batch_id = get_global_id(1);     //  the input sample of the current batch
    const int batch_size = get_global_size(1); // the size of a batch
//    const int group_size = get_local_size(0);
//    const int lid = get_local_id(0);
    // calculate the weighted input for cell gid,
    float sum = 0; // biases[gid];
    for(int i = 0; i < input_width; i++) {
        sum += weights[gid * input_width + i] * input[offset + batch_id * batch_size + i];
    }
    sum = exp(sum);
    output[gid + batch_size * batch_id] = sum;
    barrier(CLK_GLOBAL_MEM_FENCE);
    float sumExp = 0;
    for(int i = 0; i < output_width; i++) { // should be a column sum for this batch
        sumExp += output[i + batch_size * batch_id]; // this is wrong
    }
    // should probably do this reduction properly
//    exponents[gid * batch_size + batch_id] = sum;
//    barrier(CLK_LOCAL_MEM_FENCE);
//    for(int i = group_size/2; i> 2; i >> 1) {
//        if(lid < i) {
//            exponents[lid] += exponents[lid + i];
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
    output[gid + batch_size * batch_id] = sum / sumExp;
}
__kernel void softmax_layer_backward(
    __global const float* y_pred, // the probabilities, `output` in the forward step
    __global const float* x_train,
    __global const float* y_true,
    __global float* weights,
    __global float* deltas,
    const int x_width,
    const float lr, // the learning rate
    const float reg // the regularization value
) {
    // Diff is p_x-1 if y_true matches
    const int gid = get_global_id(0); // the output cell idx
    const int output_width = get_global_size(0);
    // for each W of this cell
 //   for(int i = 0; i < o)
    float diff = y_pred[gid] - 1*(y_true[gid] == 1.0);
    float dW = reg; // this needs to be over the transpose: dW = X.T * W 
    for(int i = 0; i < x_width; i++) {
        dW += diff * x_train[i];
    }
    deltas[gid] = dW;
    weights[gid] -= lr * dW;
}