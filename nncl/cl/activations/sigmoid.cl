// sigmoid

float activation_sigmoid(float x) {
    return 1 / (exp(-x) + 1);
}
float derivative_sigmoid(float x) {
    float fx = activation_sigmoid(x);
    return fx * (1-fx);
}