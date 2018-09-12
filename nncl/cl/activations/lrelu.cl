// lrelu

${dtype} activation_lrelu(${dtype} x) {
    if (x < 0)
        return 0.01*x;
    return x;
}
${dtype} derivative_lrelu(${dtype} x) {
    if (x < 0)
        return 0.01;
    return 1;
}