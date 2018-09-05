import numpy as np
from pyopencl import cltypes, clrandom, Program, CommandQueue


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
        for l in net.layers[-1::-1]:
            buf = l.backward(err, x_data, y_data, self.lr, self.reg)


class Anneal(Optimizer):
    def __init__(self, queue: CommandQueue, cooling_rate=0.001, starting_temp=1000, max_updates=100, sigma=0.66):
        self.cooling_rate = cltypes.float(cooling_rate)
        self.temperature = cltypes.float(starting_temp)
        self.max_updates = cltypes.float(max_updates)
        self.candidate_weights = []
        self.queue = queue
        self.sigma = sigma
        ctx = queue.context
        self.ctx = ctx
        self.rgen = clrandom.PhiloxGenerator(ctx)
        self.add_krnl = self.make_add_kernel()

    def make_add_kernel(self):
        src = """
        __kernel void add(__global float* a, __global const float* b) {
            // modify a such that a = a+b
            const int gid = get_global_id(0);
            a[gid] += + b[gid];
        } 
        """
        prog = Program(self.ctx, src).build()
        return prog.add

    def __call__(self, net, err, x_data, y_data, batch_number):
        # perturb the weights of each layer randomly and evaluate
        # we need a copy of the initial layers weights
        evts = []
        for i in net.layers:
            if i.is_training:
                copy = i.weights.copy()
                self.candidate_weights.append(copy)
                # array of changes to apply to candidate weights
                changes = self.rgen.normal(mu=0, sigma=self.sigma, shape=copy.shape, dtype=copy.dtype)
                evts.append(self.add_krnl(self.queue, copy.size, None, copy.data, changes.data))

        for e in evts:
            e.wait()
        for lidx, l in enumerate(net.layers):
            l()
        if np.exp(err / self.temperature) - np.random.rand() > 0:
            pass
