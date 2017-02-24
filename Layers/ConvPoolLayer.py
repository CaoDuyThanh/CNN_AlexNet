import theano
import theano.tensor
import numpy
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

class ConvPoolLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data
                 inputShape,            # Shape of input = [batch size, channels, rows, cols]
                 filterShape,           # Shape of filter = [number of filters, channels, rows, cols]
                 poolingShape = (2, 2), # Shape of pooling (2, 2) default
                 W = None
                 ):
        # Set parameters
        self.Input = input
        self.InputShape = inputShape
        self.FilterShape = filterShape
        self.PoolingShape = poolingShape

        # Create shared parameters for filters
        if W is None:
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(
                        low = -1.0,
                        high = 1.0,
                        size = self.FilterShape
                    ),
                    dtype = theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        convLayer = conv2d(
            input = self.Input,
            input_shape = (self.InputShape),
            filters = self.W,
            filter_shape = self.FilterShape,
        )

        self.poolLayer = pool_2d(
            input = convLayer,
            ds = self.PoolingShape,
            ignore_border = True
        )

    def Params(self):
        return [self.W]


    def Output(self):
        return self.poolLayer