import pyopencl as cl
from pyopencl import cltypes, CommandQueue, array
from nncl.initializer import Initializer, GlorotUniformInitializer, ZeroInitializer

mf = cl.mem_flags


class Layer:
    name = "AbstractLayer"
    def __init__(self, ctx, queue: CommandQueue, units,
                 weight_initializer: Initializer = GlorotUniformInitializer,
                 bias_initializer: Initializer = ZeroInitializer,
                 activation='linear', batch_size=64):
        self.units = units
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.ctx = ctx
        self.queue = queue
        self.is_training = True
        self.queue = queue
        self.batch_size = batch_size
        self.src = ""
        with open(f'../nncl/cl/activations/{self.activation}.cl', 'r') as infile:
            self.src += infile.read() + "\n"
        self.make_prog()

    def init(self, input_width):
        self.input_width = input_width

        if not isinstance(self.weight_initializer, Initializer):
            weight_initializer = self.weight_initializer(self.input_width, self.units)
        if not isinstance(self.bias_initializer, Initializer):
            bias_initializer = self.bias_initializer()
        self.weights_buf = weight_initializer((self.units, self.input_width))
        self.weights = array.to_device(self.queue, self.weights_buf)
        self.bias_buf = bias_initializer((self.units, 1))
        self.bias = array.to_device(self.queue, self.bias_buf)
        # should probably make this 2d so it can have dimensions (output_width, batch_size)
        self.output = array.zeros(self.queue, (self.units, self.batch_size), order='', dtype=cltypes.float)
        self.deltas = array.zeros(self.queue, self.units, dtype=cltypes.float)
        self.input_width = cltypes.uint(self.input_width)
        self.output_width = cltypes.uint(self.units)
        self.activation = self.activation
        return self.units

    def make_prog(self):
        raise ValueError("Please use an actual layer type")

    def get_bias(self):
        return self.bias.get()

    def get_weights(self):
        return self.weights.get()

    def get_output(self):
        return self.output.get()

    def forward(self, input: array.Array, weights=None):
        return self(input, weights)

    def __call__(self, input: array.Array, offset, weights: array.Array = None) -> array.Array:
        if weights is None:
            weights = self.weights
        elif weights.shape != self.weights.shape:
            raise ValueError("Invalid custom weights shape")
        self.forward_krnl(self.queue, (self.output_width, self.batch_size), None,
                          input,
                          weights.data,
                          self.bias.data,
                          self.output.data,
                          self.input_width,
                          offset
                          ).wait()
        return self.output

    def backward(self, err, x_train: cl.Buffer, y_true: cl.Buffer, lr: float, reg: float):
        if not self.is_training:
            return self.deltas
        self.backward_krnl(self.queue, (self.input_width,), None,
                           self.output.data,
                           x_train,
                           y_true,
                           self.weights.data,
                           lr,
                           reg
                           ).wait()
        return self.deltas
