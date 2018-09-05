// native tanh
// might be faster idk

float activation_native_tanh(float x) {
    return (native_exp(2*x) - 1) / (native_exp(2*x) + 1);
}

float derivative_native_tanh(float x) {
    return 1 - pow(activation_native_tanh(x), 2);
}