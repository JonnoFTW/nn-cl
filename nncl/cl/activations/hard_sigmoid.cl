// hard sigmoid

float activation_hard_sigmoid(float x) {
    return clamp(x * 0.2 + 0.5, 0.0,1.0);
}
float derivative_hard_sigmoid(float x) {
    if (x > 2.5 || x < -2.5)
        return 0;
    return 0.2;
}
