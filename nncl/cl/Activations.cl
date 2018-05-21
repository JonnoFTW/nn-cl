

// tanh

float activation_tanh(float x) {
    return tanh(x);
}
float derivative_tanh(float x) {
    return 1 - pow(tanh(x),2);
}

// hard sigmoid

float activation_hard_sigmoid(float x) {
    return clamp(x * 0.2 + 0.5, 0.0,1.0);
}
float derivative_hard_sigmoid(float x) {
    if (x > 2.5 || x < -2.5)
        return 0;
    return 0.2;
}

// sigmoid

float activation_sigmoid(float x) {
    return 1 / (exp(-x) + 1);
}
float derivative_sigmoid(float x) {
    float fx = activation_sigmoid(x);
    return fx * (1-fx);
}

// linear

float activation_linear(float x) {
    return x;
}
float derivative_linear(float x) {
    return 1;
}

// relu

float activation_relu(float x) {
    return max(0.0f, x);
}
float derivative_relu(float x) {
    if (x < 0)
        return 0;
    return 1;
}

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