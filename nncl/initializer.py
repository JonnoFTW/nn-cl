import numpy as np
from nncl.util import get_type

class Initializer:
    def __init__(self, *args):
        raise NotImplementedError("Please use an actual initializer")

    def _make_array(self, *args, **kwargs):
        raise NotImplementedError("Please use a subclass")

    def __call__(self, *args, **kwargs):
        return self._make_array(*args, **kwargs).astype(get_type())


class DistributionInitializer(Initializer):

    def __init__(self, low=0, high=1):
        """
        Base class for an initialiser with a distribution between [low,high]
        :param low:
        :param high:
        """
        self.low = low
        self.high = high


class ConstantInitializer(Initializer):
    """
    Initializer that outputs array of a constant value
    """

    def __init__(self, value: float = 0):
        self.value = value

    def _make_array(self, count):
        return np.full(count, self.value)


class ZeroInitializer(ConstantInitializer):
    """
    Initializer of zeros
    """
    pass


class OnesInitializer(ConstantInitializer):
    """
    Initializer for only ones
    """

    def __init__(self):
        self.value = 1


class UniformInitializer(DistributionInitializer):
    """
    A uniform distribution in low to high
    """

    def _make_array(self, shape):
        return np.random.uniform(self.low, self.high, shape)


class HeUniformInitializer(UniformInitializer):
    def __init__(self, input_width):
        limit = np.sqrt(6. / input_width)
        self.low, self.high = -limit, limit


class LecunUniformInitializer(UniformInitializer):
    def __init__(self, input_width):
        limit = np.sqrt(3. / input_width)
        self.low, self.high = -limit, limit


class GlorotUniformInitializer(UniformInitializer):
    def __init__(self, input_width, output_width):
        self.low = -(np.sqrt(6) / np.sqrt(input_width + output_width))
        self.high = np.negative(self.low)


class NormalInitializer(Initializer):
    """
    Initializer for a normal distribution
    """

    def __init__(self, center=0, stddev=0.05):
        """
        :param center:
        :param scale:
        """
        self.center = center
        self.stddev = stddev

    def _make_array(self, count):
        return np.random.normal(self.center, self.stddev, count)


class GlorotNormalInitializer(NormalInitializer):
    def __init__(self, input_width, output_width):
        self.center = 0
        self.stddev = np.sqrt(2. / (input_width + output_width))


class HeNormalInitializer(NormalInitializer):
    def __init__(self, input_width):
        self.center = 0
        self.stddev = np.sqrt(2. / input_width)


class LecunNormalInitializer(NormalInitializer):
    def __init__(self, input_width):
        self.center = 0
        self.stddev = np.sqrt(1. / input_width)
