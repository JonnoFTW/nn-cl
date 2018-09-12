// sigmoid

${dtype} activation_sigmoid(${dtype} x) {
    return 1 / (exp(-x) + 1);
}
${dtype} derivative_sigmoid(${dtype} x) {
    ${dtype} fx = activation_sigmoid(x);
    return fx * (1-fx);
}