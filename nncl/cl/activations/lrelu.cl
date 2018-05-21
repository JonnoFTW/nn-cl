// lrelu

float activation_lrelu(float x) {
    if (x < 0)
        return 0.01*x;
    return x;
}
float derivative_lrelu(float x) {
    if (x < 0)
        return 0.01;
    return 1;
}