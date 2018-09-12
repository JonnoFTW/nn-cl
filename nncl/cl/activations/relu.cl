// reludata

${dtype} activation_relu(${dtype} x) {
    return max(0.0f, x);
}
${dtype} derivative_relu(${dtype} x) {
    return step(0.0f, x);
}
