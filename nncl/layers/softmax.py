import pyopencl as cl
from nncl.layers.layer import Layer


class Softmax(Layer):
    name = "Softmax"

    def __init__(self, *args, **kwargs):
        kwargs['activation'] = 'softmax'
        super().__init__(*args, **kwargs)

    def backward(self, err, x_train: cl.Buffer, y_true: cl.Buffer, lr: float, reg: float):
        self.backward_krnl(self.queue, (self.input_width,), None,
                           self.output.data,
                           x_train,
                           y_true,
                           self.weights.data,
                           self.bias.data.
                           self.deltas.data,
                           self.output_width,
                           lr,
                           reg
                           ).wait()
        return self.deltas

    def make_prog(self):
        with open('../nncl/cl/layers/DenseSoftmax.cl', 'r') as infile:
            self.src += infile.read().replace('${dtype}', self.dtype_str) + "\n"

        self.prog = cl.Program(self.ctx, self.src).build()
        self.forward_krnl = self.prog.softmax_layer_forward
        self.backward_krnl = self.prog.softmax_layer_backward
