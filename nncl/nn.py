import numpy as np
import pyopencl as cl
from pyopencl import array, cltypes
from tqdm import tqdm
from typing import List
import humanize
from nncl.layers.layer import Layer
from nncl.losses import Loss
from nncl.optimizers import Optimizer
from nncl.util import get_type, get_type_repl

mf = cl.mem_flags

dtype = get_type()
dtype_str = get_type_repl()


class Network:
    def __init__(self, input_size, ctx=None, batch_size=64):
        self.input_size = input_size
        if ctx is None:
            self.ctx = cl.Context([cl.get_platforms()[0].get_devices()[1]])
        else:
            self.ctx = ctx
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.queue = cl.CommandQueue(ctx)
        self.layers: List[Layer] = []

        with open('../nncl/cl/utils/shuffle.cl', 'r') as infile:
            src = infile.read().replace('${dtype}', dtype_str)
            shuffle_prog = cl.Program(self.ctx, src).build()
        self.shuffle_krnl = shuffle_prog.shuffle_data

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build(self):
        # go through the layers and set input/output dimensions appropriately
        if len(self.layers) == 0:
            raise ValueError("You need to add layers before you can build the model")
        units = self.input_size
        for l in self.layers:
            units = l.init(units)

    def summary(self):
        for idx, l in enumerate(self.layers):
            print(f"Layer {idx} {l.name}")
            print(f"\tInputs: {l.input_width}")
            print(f"\tUnits: {l.units}")
            print(f"\tActivation: {l.activation}")

    def forward(self, buf: array.Array, verbose: bool = False):
        # put x in the buffer
        size = self.layers[0].input_width
        # can probably do better here
        # this only works on pocl because they didn't implement CL_MISALIGNED_SUB_BUFFER_OFFSET 
        #  buf = x.get_sub_region(size * idx, size)
        input_np = buf.get()
        for idx, l in enumerate(self.layers):
            l.inputs = input_np.copy()
            if verbose:
                print(f"Layer {idx}")
                print(f"Input Batch: rows={l.batch_size} samples cols={l.input_width} features \n", input_np)
            buf = l(buf)
            output = buf.get()
            if verbose:
                weights = l.get_weights()
                bias = l.get_bias()
                print(f"\nWeights: (rows={l.units} units, cols={l.input_width} inputs)\n", weights)
                # print("Biases:\n", bias)
                print(f"\nOutput: (rows={l.batch_size} batch samples cols={l.units} units)\n", output)
                expected = (np.dot(weights, input_np.T) + bias).T
                if l.activation == 'relu':
                    expected = np.clip(expected, 0, a_max=None)
                elif l.activation == 'sigmoid':
                    expected = 1 / (np.exp(-expected) + 1)
                elif l.activation == 'softmax':
                    exps = np.exp(expected)
                    expected = exps / exps.sum(axis=1)[:, None]
                print("Expected:\n", expected)
            input_np = output

        # output is the output of the last layer
        return buf

    def read_only_arr(self, numbytes):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, numbytes)

    def shuffle(self, x_data, y_data, samples, x_fields, y_fields, swaps_g=None):
        """
        Shuffles the rows of x_data, y_data. Switches based on the contents of
        swaps_g if provided. If an odd number of rows are passed, then one row
        will not be swapped

        :param x_data:
        :param y_data:
        :param samples:
        :param x_fields:
        :param y_fields:
        :param swaps_g:
        :return:
        """
        if swaps_g is None:
            swaps_np = np.arange(samples, dtype=cltypes.uint)
            np.random.shuffle(swaps_np)
            swaps_g = array.to_device(self.queue, swaps_np, allocator=self.read_only_arr)
        # print("swaps:", swaps_g.get())
        e1 = self.shuffle_krnl(self.queue, (x_fields, samples // 2), None, x_data, swaps_g.data)
        e2 = self.shuffle_krnl(self.queue, (y_fields, samples // 2), None, y_data, swaps_g.data)
        e1.wait()
        e2.wait()

    def train(self, epochs: int,
              loss: Loss,
              optimizer: Optimizer,
              x_train,
              y_train,
              x_test,
              y_test,
              x_validation=None,
              y_validation=None,
              batch_size: int = 1,
              shuffle: bool = True,
              validation_pct=None,
              validation_method='cross-validation',
              callbacks=[]):
        """

        :param epochs: number of epochs to run
        :param loss:  a loss function
        :param optimizer: the optimizer to use
        :param x_train: a 2D array of shape (rows, features)
        :param y_train: a 2d array of shape (rows, output features),
                output_features is the number of values we want to predict
        :param x_test: testing data inputs
        :param y_test: testing data true values
        :param validation_method: a string to determine which validation  method to use: 'holdout','cross-validation'
        :return: None

        For example, our input might be:
        x_train = [
            [0,1,1],
            [0,2,1],
            [1,2,1],
            [0,3,4],
        ]
        That is 4 rows with 3 features each, we might do a binary classification on this:
        y_train = [
            [0,1],
            [0,1],
            [1,0],
            [0,1]
        ]
        That is, each training input maps to one of these
        All this will be copied to the device

        Validation methods are:

        1. Specify x_validation,y_validation and the same provided dataset will be used to validate every epoch
        2. Specify validation_pct to determine how much of the training set will be set aside as validation.
           Specify validation_method to determine which method to use:
            * holdout: the same subset of x_train is used to validate each epoch
            * cross-validation: at the start of each epoch a random sample of x_train/y_train is set aside


        """

        if validation_pct is not None and x_validation is not None and y_validation is not None:
            raise ValueError("Please set either validation_pct or (x_validation,x_validation)")
        if x_validation is not None != x_validation is not None:
            raise ValueError("Please set both (x_validation and y_validation)")

        x_train = x_train.astype(dtype)
        y_train = y_train.astype(dtype)

        if validation_pct:

            # slice off the last validation_ct from x_train,y_train
            if 0 <= validation_pct < 1:
                training_samples = int(x_train.shape[0] * (1 - validation_pct))
                validation_samples = int(x_train.shape[0] * validation_pct)
                if validation_method == 'holdout':
                    print(f"Holding out last {validation_samples} samples of training data for validation")
                    x_train = x_train[:training_samples]
                    y_train = y_train[:training_samples]

                    x_validation = x_train[training_samples:]
                    y_validation = y_train[training_samples:]
                    x_val_gpu = array.to_device(self.queue, x_validation)
                    y_val_gpu = array.to_device(self.queue, y_validation)
                elif validation_method == 'cross-validation':
                    print(f"Using cross-validation on last {validation_samples}")
                else:
                    raise ValueError("Invalid validation method")
                validation_user = False

            else:
                raise ValueError("Validation_pct must be in range 0 <= val% < 1")
        elif x_validation is not None and y_validation is not None:
            print("User provided validation")
            x_validation = x_validation.astype(dtype)
            y_validation = y_validation.astype(dtype)
            x_val_gpu = array.to_device(self.queue, x_validation)
            y_val_gpu = array.to_device(self.queue, y_validation)
            validation_samples = len(x_validation)
            training_samples = x_train.shape[0]
            validation_user = True
        else:
            training_samples = x_train.shape[0]
        if len(x_train) != len(y_train):
            raise ValueError("X and Y for test/train must be same length")
        if training_samples % batch_size != 0:
            raise ValueError("Training dataset must have rows divisible by batch size")

        input_features = cltypes.uint(x_train.shape[1])
        output_features = cltypes.uint(y_train.shape[1])
        if input_features != self.layers[0].input_width:
            raise ValueError(
                f"Input features (provided={input_features}) must be the same as layer_0 input width (required={self.layers[0].input_width})")
        # Just copy all training and all testing data to the device
        for dn, ds in ("x_train", x_train), ("y_train", y_train), ("x_validation", x_validation), (
                "y_validation", y_validation):
            try:
                print("{}\n\tsize={}\n\tshape={}".format(dn, humanize.naturalsize(ds.nbytes), ds.shape))
            except AttributeError:
                pass

        # x_train_gpu = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x_train)
        x_train_gpu = array.to_device(self.queue, x_train)
        y_train_gpu = array.to_device(self.queue, y_train)

        # should probably check that our data won't exceed available device memory,
        # transparently queue up more data once it's been used
        losses = {
            'batch': [],
            'validation': [],
            'testing': []
        }
        for i in tqdm(range(epochs), desc='Epoch: ', position=0):
            # shuffle the rows
            if shuffle:
                self.shuffle(x_train_gpu.data, y_train_gpu.data, training_samples, input_features, output_features)
            for idx in tqdm(range(training_samples // batch_size), desc='Batch: ', position=1, unit=' batch'):
                idx = cltypes.uint(idx)
                # idx here is the batch number

                batch_x_gpu = x_train_gpu[idx * batch_size: idx * batch_size + batch_size]
                batch_y_gpu = y_train_gpu[idx * batch_size: idx * batch_size + batch_size]
                # copy all of these to the device?
                output = self.forward(batch_x_gpu, verbose=False)
                loss_val = loss.cpu(batch_y_gpu, output)
                # err = loss(batch_y_gpu, output, )
                losses['batch'].append(loss_val)
                # print(f"Mean Batch Loss={loss_val}")
                optimizer(loss, self, batch_x_gpu, batch_y_gpu)
                # if idx % 900 == 0:
                #     for c in callbacks:
                #         if c.batch_end:
                #             c(losses)
            # run the network and get error for the validation set
            # this should be a single batch of size validation_samples
            # will need to allocate specific validation arrays

            # if validation_user:
            #     # validate with user supplied validation data
            #     output = self.forward(x_val_gpu, 0)  # should probably be done as a single batch,
            #     val_loss = loss(y_val_gpu, output, 0)
            # else:
            #     # idx is the index of the validation set start position
            #     idx = len(x_train) - validation_samples
            #     output = self.forward(x_train_gpu, idx)
            #     val_loss = loss(y_train_gpu, output, idx)
            # losses['validation'].append(val_loss)
            # # collect metrics for training set
            # output = self.forward(x_test, 0)
            # test_loss = loss(y_test, output, 0)
            # losses['testing'].append(test_loss)
            for c in callbacks:
                c(losses)
        return losses
