/**
 * Densely connected layer
**/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DEBUG 0
__kernel void dense_layer_forward(
   __global const ${dtype}* input,   // input buffer
   __global const ${dtype}* weights, // layer weights
   __global const ${dtype}* biases,  // layer bias
   __global ${dtype}* output,  // output buffer, (columns are cell output, rows are batch idx)
    const int input_width,  // input width
    const int offset // where this batch's input data starts
) {
    // Calculates the activation of the cell
    // Activation is the weighted sum of the inputs
    const int unit = get_global_id(1);
    const int output_width = get_global_size(1);
    const int batch_id = get_global_id(0);     //  the index into the batch we want to do
    const int batch_size = get_global_size(0); // the size of a batch
    ${dtype} sum = biases[unit];
    const int output_idx = batch_id *  output_width + unit;
    #if DEBUG
        printf("DenseKrnl unit=%d/%d batch=%d/%d output_idx=%d\n", unit, output_width-1, batch_id, batch_size-1, output_idx);
    #endif
    for(int i = 0; i < input_width; i++) {
        const ${dtype} weight = weights[unit * input_width + i];
        const int input_idx = offset + (batch_id * input_width) + i;
        const ${dtype} input_val = input[input_idx];
        #if DEBUG
            printf("\tW_%d,%d=%.3f x input=%.3f = %.3f (input_idx=%d)\n",unit,i, weight, input_val, weight*input_val, input_idx);
        #endif
        sum += weight * input_val;
    }
    #if DEBUG
        printf("\tTotal=%.3f\n", sum);
    #endif
    output[output_idx] = ${activation}(sum);
}
__kernel void dense_layer_backward(
    __global const ${dtype}* output, // this layer's outputs from the forward pass
    __global ${dtype}* weights, // current layers weights to be updated
    __global const ${dtype}* expected, // the expected output of this layer
    __global ${dtype}* errors, // the error for each batch
    const int input_width
) {
    const int unit = get_global_id(0); // the index of the unit
    const int output_width = get_global_size(0);
    const int batch_id = get_global_id(1);     //  the index into the batch we want to do
    const int batch_size = get_global_size(1); // the size of a batch
    const int output_idx = output_width * batch_id + unit;
    const ${dtype} output_value = output[output_idx];
    const ${dtype} expected_value = expected[output_idx];
    const ${dtype} error = (expected_value - output_value) * ${derivative}(output_value);
//    printf("unit: %d error: %.3f expected: %.3f unit_output: %.3f\n", unit, error, expected_value, output_value);

    errors[output_idx] = error;

}

