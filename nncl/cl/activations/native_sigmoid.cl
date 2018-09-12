// native_sigmoid

${dtype} activation_native_sigmoid(${dtype} x) {
    return 1 / (native_exp(-x) + 1);
}
${dtype} derivative_native_sigmoid(${dtype} x) {
    ${dtype} fx = activation_sigmoid(x);
    return fx * (1-fx);
}