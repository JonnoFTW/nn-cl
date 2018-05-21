import numpy as np


class Initializer:
    def __init__(self, *args):
        raise NotImplementedError("Please use an actual initializer")
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Please use an actual initializer")

class DistributionInitializer(Initializer):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def __call__(self, count):
        raise NotImplementedError("Please use a subclass")


class UniformInitializer(DistributionInitializer):
    def __call__(self, shape):
        return np.random.uniform(self.low, self.high, shape)


class ZeroInitializer(Initializer):
    def __call__(self, count):
        return np.zeros(count)


class OnesInitializer(Initializer):
    def __call__(self, count):
        return np.ones(count)


class NormalInitializer(Initializer):
    def __init__(self, center, scale):
        self.center = center
        self.scale = scale

    def __call__(self, count):
        return np.random.normal(self.center, self.scale, count)


class GlorotUniformInitializer(UniformInitializer):
    def __init__(self, inputs, outputs):
        self.low = -(np.sqrt(6) / np.sqrt(inputs + outputs))
        self.high = np.negative(self.low)

