// native tanh
// might be faster idk

${dtype} activation_native_tanh(${dtype} x) {
    return (native_exp(2*x) - 1) / (native_exp(2*x) + 1);
}

${dtype} derivative_native_tanh(${dtype} x) {
    return 1 - pow(activation_native_tanh(x), 2);
}