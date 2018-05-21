import pyopencl as cl
from pyopencl import cltypes, CommandQueue
from .initializer import Initializer, GlorotUniformInitializer
import numpy as np
from glob import glob
from mako.template import Template

mf = cl.mem_flags


class Layer:
    def __init__(self, ctx, queue: CommandQueue, input_width, output_width,
                 initializer: Initializer = GlorotUniformInitializer,
                 activation='linear'):
        self.weights_buf = initializer(input_width, output_width)((output_width, input_width)).astype(
            dtype=cltypes.float)
        self.weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.weights_buf)
        self.output_buf = np.zeros(output_width, dtype=cltypes.float)
        self.output = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.output_buf)
        self.input_width = cltypes.uint(input_width)
        self.output_width = cltypes.uint(output_width)
        self.activation = activation
        self.bias = 0
        self.ctx = ctx
        self.queue = queue
        self.make_prog()

    def make_prog(self):
        src = ""
        for f in glob('../nncl/cl/*.cl'):
            with open(f, 'r') as infile:
                src += infile.read() + "\n"
        src = Template(src).render(activation='activation_' + self.activation,
                                   derivative='derivative_' + self.activation)
        # print(src)
        self.prog = cl.Program(self.ctx, src).build()

    def get_weights(self):
        cl.enqueue_copy(self.queue, self.weights_buf, self.weights).wait()
        return self.weights_buf

    def get_output(self):
        cl.enqueue_copy(self.queue, self.output_buf, self.output).wait()
        return self.output_buf


class Dense(Layer):
    def __call__(self, input: cl.Buffer):
        """
        __global float* input,   // input buffer
        __global float* weights, // layer weights
        __global float* output,  // output buffer
        const int input_width   // input width
        const int output_width
        :param input:
        :param input_width:
        :return:
        """
        self.prog.dense_layer_forward(self.queue, (self.output_width,), None,
                                      input, self.weights, self.output, self.input_width
                                      ).wait()
        return self.output
