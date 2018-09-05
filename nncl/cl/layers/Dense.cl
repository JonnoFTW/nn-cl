/**
 * Densely connected layer
**/
#define DEBUG 0
__kernel void dense_layer_forward(
   __global const float* input,   // input buffer
   __global const float* weights, // layer weights
   __global const float* biases,  // layer bias
   __global float* output,  // output buffer, (columns are cell output, rows are batch idx)
    const int input_width,  // input width
    const int offset // where this batch's input data starts
) {
    // Calculates the activation of the cell
    // Activation is the weighted sum of the inputs
    const int gid = get_global_id(0);
    const int output_width = get_global_size(0);
    const int batch_id = get_global_id(1);     //  the index into the batch we want to do
    const int batch_size = get_global_size(1); // the size of a batch
    float sum = biases[gid];
    const int output_idx = output_width * batch_id + gid;
    #if DEBUG
        if(gid == 0 && batch_id==0) {
            printf("Inputs for batch are:\n");
            for(int i=0; i< batch_size;i++) {
                printf("\t%d=%.3f\n", i, input[i+offset+batch_id]);
            }
        }
        printf("DenseKrnl unit=%d/%d batch=%d/%d output_idx=%d\n", gid, output_width-1, batch_id, batch_size-1, output_idx);
    #endif
    for(int i = 0; i < input_width; i++) {
        const float weight = weights[gid * input_width + i];
        const int input_idx = i + offset + batch_id;
        const float input_val = input[input_idx];
        #if DEBUG
            printf("\tW_%d,%d=%.3f x input=%.3f = %.3f (input_idx=%d)\n",gid,i, weight, input_val, weight*input_val, input_idx);
        #endif
        sum += weight * input_val;
    }
    #if DEBUG
        printf("\tTotal=%.3f\n", sum);
    #endif
    output[output_idx] = ${activation}(sum);
}
__kernel void dense_layer_backward(
    __global const float* input, // input is the errors
    __global float* weights, // current layers weights to be updated
    __global float* output, // the gradients of this layer
    const int input_width
) {
    const int gid = get_global_id(0);
    const float error = ${derivative}(input[gid]);

    // calculate output as transpose

}

