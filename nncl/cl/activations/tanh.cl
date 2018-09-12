// tanh

${dtype} activation_tanh(${dtype} x) {
    return tanh(x);
}
${dtype} derivative_tanh(${dtype} x) {
    return 1 - pow(tanh(x),2);
}

