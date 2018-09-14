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
                 sigma: float = 0.02):
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
        self.add_krnl = self.make_add_kernel()

    def make_add_kernel(self):
        src = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void add_rand(__global const ${dtype}* a,
                       __global float* out, 
                       const float seed) {
    // modify out such that out = a+ random
    const int gid = get_global_id(0);
    float ptr = 0.0f;
    float randval =  fract(sin(gid*112.9898f + seed*237.212f) * 43758.5453f, &ptr);
    const float min = 0.;
    const float max = 1.;
    const float scaledMax =  0.005;
    const float scaledMin = -0.005;
    const float scaled = (scaledMax-scaledMin)*(randval-min)/(max-min)+scaledMin;
    out[gid] = a[gid] + scaled;
} 
        """.replace('${dtype}', dtype_str)
        prog = Program(self.ctx, src).build()
        return prog.add_rand

    def __call__(self, net, err, x_data, y_data, idx):
        # perturb the weights of each layer randomly and evaluate
        # we need a copy of the initial layers weights
        temperature = self.temperature
        candidates: array.Array = [array.empty_like(l.weights, self.queue) for l in net.layers]
        currents: array.Array = [l.weights for l in net.layers]
        for u in range(self.max_updates):

            evs = []
            for current, candidate in zip(currents, candidates):
                # add noise to candidate weights
                evs.append(self.add_krnl(self.queue, (candidate.size,), None,
                                         current.data, candidate.data, cltypes.float(np.random.randn())))
            for e in evs:
                e.wait()
            # get the new output using these weights
            offset = cltypes.int(net.batch_size * net.layers[0].input_width * idx)
            buf = x_data
            for candidate, l in zip(candidates, net.layers):
                buf = l.forward(buf, offset, weights=candidate.data)
                offset = cltypes.int(0)
            output = net.layers[-1].output
            candidate_err = self.loss.cpu(y_data, output, idx)
            if candidate_err == np.nan:
                continue
            err_delta = err - candidate_err
            accept = candidate_err < err  or np.exp(err_delta / temperature) - np.random.rand() > 0
            # accept the candidate solution

            if accept:
                # print(f"Accepting {candidate_err} over {err},{'worse' if err_delta < 0 else 'better'}")
                err = candidate_err
                currents = candidates
            # else:
            #     print(f"Rejecting candidate {candidate_err}")
            temperature *= self.cooling_rate
        for l, c in zip(net.layers, currents):
            l.weights = c
