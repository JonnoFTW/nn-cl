// relu

float activation_relu(float x) {
    return max(0.0f, x);
}
float derivative_relu(float x) {
    if (x < 0)
        return 0;
    return 1;
}
