/**
 * Densely connected layer
**/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DEBUG 0
__kernel void layer_forward(
   __global const ${dtype}* input,   // input buffer
   __global const ${dtype}* weights, // layer weights
   __global const ${dtype}* biases,  // layer bias
   __global ${dtype}* output,  // output buffer, (columns are cell output, rows are batch idx)
    const int input_width  // input width
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
        const int input_idx = (batch_id * input_width) + i;
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
