// tanh

float activation_tanh(float x) {
    return tanh(x);
}
float derivative_tanh(float x) {
    return 1 - pow(tanh(x),2);
}




