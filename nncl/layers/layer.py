import sys

import pyopencl as cl
import numpy as np
from mako.template import Template
from pyopencl import cltypes, CommandQueue, array, Program
from nncl.initializer import Initializer, GlorotUniformInitializer, ZeroInitializer
from nncl.util import get_type, get_type_repl

mf = cl.mem_flags

dtype = get_type()
dtype_str = get_type_repl()


class Layer:
    name = "AbstractLayer"
    layer_fname = None

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
        for fname in [f'../nncl/cl/activations/{self.activation}.cl',
                      self.layer_fname,
                      '../nncl/cl/layers/gradient.cl']:
            with open(fname, 'r') as infile:
                self.src += infile.read() + "\n"
        self.src = Template(self.src).render(
            activation='activation_' + self.activation,
            derivative='derivative_' + self.activation,
            dtype=self.dtype_str
        )
        try:
            self.prog = Program(self.ctx, self.src).build()
        except cl.cffi_cl.RuntimeError as e:
            # print(self.src, flush=True)
            print(e, file=sys.stderr, flush=True)
            exit(1)
        self.forward_krnl = self.prog.layer_forward

        self._grad_krnl = self.prog.get_gradients

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

    def forward(self, input: array.Array, weights=None):
        return self(input, weights)

    def __call__(self, input: array.Array, weights=None) -> array.Array:
        if weights is None:
            weights = self.weights.data
        self.forward_krnl(self.queue, (self.batch_size, self.output_width), None,
                          input.data,
                          weights,
                          self.bias.data,
                          self.output.data,
                          self.input_width
                          ).wait()
        return self.output

    def backward(self, y_errors, lr):
        if not self.is_training:
            return
        # just do it ont he cPu lmao
        # weight deltas are the input to this layer dot grads
        grads = cl.array.to_device(self.queue, y_errors.T.get().dot(self.inputs))
        # apply derivative activation to get deltas for weights
        # and then multiply by learning rate

        ev = self._grad_krnl(self.queue, (grads.size,), None,
                             grads.data,
                             self.weights.data, lr
                             )
        return ev
