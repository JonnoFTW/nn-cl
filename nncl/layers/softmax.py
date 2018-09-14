import pyopencl as cl
from nncl.layers.layer import Layer


class Softmax(Layer):
    name = "Softmax"
    layer_fname = '../nncl/cl/layers/DenseSoftmax.cl'

    def __init__(self, *args, **kwargs):
        kwargs['activation'] = 'softmax'
        super().__init__(*args, **kwargs)
