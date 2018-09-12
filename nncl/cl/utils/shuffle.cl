__kernel void shuffle_data(
    __global ${dtype}* data, // array to be shuffled
    __constant uint* swaps  // array of swaps to perform
) {
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    const int num_cols = get_global_size(0);

    const int idx1 = num_cols * swaps[2*row] + col;
    const int idx2 = num_cols * swaps[2*row+1] + col;
    // if(col==0)  printf("Swapping %d with %d\\n", swaps[2*row], swaps[2*row+1]);
    float tmp = data[idx2];
    data[idx2] = data[idx1];
    data[idx1] = tmp;
}