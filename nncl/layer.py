import pyopencl as cl
from pyopencl import cltypes, CommandQueue, array
from .initializer import Initializer, GlorotUniformInitializer
import numpy as np
from mako.template import Template

mf = cl.mem_flags


class Layer:
    def __init__(self, ctx, queue: CommandQueue, input_width, output_width,
                 initializer: Initializer = GlorotUniformInitializer,
                 activation='linear'):
        self.weights_buf = initializer(input_width, output_width)((output_width, input_width)).astype(
            dtype=cltypes.float)
        self.weights = array.to_device(queue, self.weights_buf)
        # should probably make this 2d so it can have dimensions (output_width, batch_size)
        self.output = array.zeros(queue, output_width, dtype=cltypes.float)
        self.input_width = cltypes.uint(input_width)
        self.output_width = cltypes.uint(output_width)
        self.activation = activation
        self.bias = 0
        self.ctx = ctx
        self.queue = queue
        self.src = ""
        with open(f'../nncl/cl/activations/{self.activation}.cl', 'r') as infile:
            self.src += infile.read() + "\n"
        self.make_prog()

    def make_prog(self):
        raise ValueError("Please use an actual layer type")

    def get_weights(self):
        return self.weights.get()

    def get_output(self):
        return self.output.get()

    def __call__(self, input: array.Array) -> array.Array:
        self.forward(self.queue, (self.output_width,), None,
                     input,
                     self.weights.data,
                     self.output.data,
                     self.input_width
                     ).wait()
        return self.output

    def backward(self, input: cl.Buffer):
        self.backward(self.queue, (self.input_width,), None, input).wait()


class Softmax(Layer):
    def __init__(self, ctx, queue: CommandQueue, input_width, output_width,
                 initializer: Initializer = GlorotUniformInitializer):
        super().__init__(ctx, queue, input_width, output_width, initializer=initializer,
                         activation='softmax')
        self.forward = self.prog.softmax_layer_forward

    def make_prog(self):
        with open('../nncl/cl/layers/DenseSoftmax.cl', 'r') as infile:
            self.src += infile.read() + "\n"

        self.prog = cl.Program(self.ctx, self.src).build()
        self.forward = self.prog.softmax_layer_forward
        self.backward = self.prog.softmax_layer_backward


class Dense(Layer):

    def __init__(self, ctx, queue: CommandQueue, input_width, output_width,
                 initializer: Initializer = GlorotUniformInitializer,
                 activation='linear'):
        super().__init__(ctx, queue, input_width, output_width, initializer=initializer,
                         activation=activation)
        self.forward = self.prog.dense_layer_forward

    def make_prog(self):
        with open('../nncl/cl/layers/Dense.cl', 'r') as infile:
            self.src += infile.read() + "\n"
        self.src = Template(self.src).render(
            activation='activation_' + self.activation,
            derivative='derivative_' + self.activation
        )
        self.prog = cl.Program(self.ctx, self.src).build()
        self.forward = self.prog.dense_layer_forward
        self.backward = self.prog.dense_layer_backward
