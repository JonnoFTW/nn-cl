from mako.template import Template
from pyopencl import CommandQueue, Program

from nncl.initializer import GlorotUniformInitializer, Initializer
from nncl.layers.layer import Layer


class Dense(Layer):
    name = "Dense"
    def __init__(self, ctx, queue: CommandQueue, units,
                 initializer: Initializer = GlorotUniformInitializer,
                 activation='linear', batch_size=64):
        super().__init__(ctx, queue, units=units, weight_initializer=initializer,
                         activation=activation, batch_size=batch_size)

    def make_prog(self):
        with open('../nncl/cl/layers/Dense.cl', 'r') as infile:
            self.src += infile.read() + "\n"
        self.src = Template(self.src).render(
            activation='activation_' + self.activation,
            derivative='derivative_' + self.activation
        )
        self.prog = Program(self.ctx, self.src).build()
        self.forward_krnl = self.prog.dense_layer_forward
        self.backward_krnl = self.prog.dense_layer_backward
