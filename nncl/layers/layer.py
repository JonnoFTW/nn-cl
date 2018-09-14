import pyopencl as cl
from multiprocessing.dummy import Value
from pyopencl import cltypes, CommandQueue, array
from nncl.initializer import Initializer, GlorotUniformInitializer, ZeroInitializer
from nncl.util import get_type, get_type_repl

mf = cl.mem_flags

dtype = get_type()
dtype_str = get_type_repl()


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
        self.dtype_str = dtype_str
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
        self.output = array.zeros(self.queue, (self.batch_size, self.units), dtype=dtype)
        self.output_data = self.output.data
        self.deltas = array.zeros(self.queue, self.units, dtype=dtype)
        self.errors = array.zeros(self.queue, (self.units, self.batch_size), dtype=dtype)
        self.input_width = cltypes.uint(self.input_width)
        self.output_width = cltypes.uint(self.units)
        self.activation = self.activation
        max_output, max_batch_size = self.queue.device.max_work_item_sizes[:2]
        if self.output_width > max_output:
            raise ValueError(f"Layer output cannot exceed {max_output}, you gave {self.output_width}")
        if self.batch_size > max_batch_size:
            raise ValueError(f"Batch size cannot exceed {max_batch_size}, you gave {self.batch_size}")

        return self.units

    def make_prog(self):
        raise ValueError("Please use an actual layer type")

    def get_bias(self):
        return self.bias.get()

    def get_weights(self):
        return self.weights.get()

    def get_output(self):
        return self.output.get()

    def forward(self, input: array.Array, offset, weights=None):
        return self(input, offset, weights)

    def __call__(self, input: array.Array, offset, weights=None) -> array.Array:
        if weights is None:
            weights = self.weights.data
        self.forward_krnl(self.queue, (self.batch_size, self.output_width), None,
                          input,
                          weights,
                          self.bias.data,
                          self.output.data,
                          self.input_width,
                          offset
                          ).wait()
        return self.output_data

    def backward(self,
                 loss,
                 x_train: cl.Buffer,
                 y_true: cl.array,
                 lr: float,
                 reg: float):
        if not self.is_training:
            return self.deltas

        # compute
        #  errors
        self.backward_krnl(self.queue, (self.output_width, self.batch_size), None,
                           self.output.data,
                           self.weights.data,
                           y_true.data,
                           self.errors.data,
                           self.input_width
                           ).wait()
        return self.deltas
