import numpy as np
from pyopencl import cltypes, clrandom, Program, CommandQueue
from pyopencl import array

from nncl.util import get_type, get_type_repl

dtype = get_type()
dtype_str = get_type_repl()


class Optimizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Please use an actual optimizer")

    def __call__(self, net, err, x_data, y_data, batch_number):
        raise NotImplementedError("Please use an actual optimizer")


class SGD(Optimizer):
    def __init__(self, learing_rate=0.1, reg=0.003):
        self.learning_rate = cltypes.float(learing_rate)
        self.lr = self.learning_rate
        self.reg = cltypes.float(reg)

    def __call__(self, net, err, x_data, y_data, batch_number):
        for l in net.layers[::-1]:
            buf = l.backward(err, x_data, y_data, self.lr, self.reg)
            print("Errors:\n", l.errors.get())


class Anneal(Optimizer):
    def __init__(self, queue: CommandQueue, loss, cooling_rate: float = 0.9999, starting_temp: float = 500.,
                 max_updates: int = 100,
                 sigma: float = 0.001):
        self.cooling_rate = cooling_rate
        self.temperature = starting_temp
        self.max_updates = max_updates
        self.candidate_weights = []
        self.queue = queue
        self.sigma = sigma
        self.loss = loss
        ctx = queue.context
        self.ctx = ctx
        # self.rgen = clrandom.PhiloxGenerator(ctx)
        # self.add_krnl = self.make_add_kernel()

    def make_add_kernel(self):
        src = """
        __kernel void add(__global ${dtype}* a, __global const float* b) {
            // modify a such that a = a+b
            const int gid = get_global_id(0);
            a[gid] += + b[gid];
        } 
        """
        prog = Program(self.ctx, src).build()
        return prog.add

    def __call__(self, net, err, x_data, y_data, idx):
        # perturb the weights of each layer randomly and evaluate
        # we need a copy of the initial layers weights
        temperature = self.temperature

        for u in range(self.max_updates):
            candidate_weights = []

            # get the new output using these weights
            offset = cltypes.int(net.batch_size * net.layers[0].input_width * idx)
            buf = x_data
            for l in net.layers:
                c_arr = array.to_device(self.queue,
                    np.random.normal(0, self.sigma, l.weights.shape).astype(dtype=dtype) + l.weights.get())
                candidate_weights.append(c_arr)
                buf = l.forward(buf, offset, weights=candidate_weights[-1].data)
                offset = cltypes.int(0)
            output = net.layers[-1].output
            candidate_err = self.loss.cpu(y_data, output, idx)
            err_delta = err - candidate_err
            accept = err_delta < 0 or np.exp(err_delta / temperature) - np.random.rand() > 0
            # accept the candidate solution
            if accept:
                for cand, l in zip(candidate_weights, net.layers):
                    l.weights = cand
            temperature *= self.cooling_rate
