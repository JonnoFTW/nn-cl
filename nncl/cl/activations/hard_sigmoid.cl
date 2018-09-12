// hard sigmoid

${dtype} activation_hard_sigmoid(${dtype} x) {
    return clamp(x * 0.2 + 0.5, 0.0,1.0);
}
${dtype} derivative_hard_sigmoid(${dtype} x) {
    if (x > 2.5 || x < -2.5)
        return 0;
    return 0.2;
}
