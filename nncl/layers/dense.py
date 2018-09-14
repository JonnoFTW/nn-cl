from nncl.layers.layer import Layer


class Dense(Layer):
    name = "Dense"
    layer_fname = '../nncl/cl/layers/Dense.cl'
