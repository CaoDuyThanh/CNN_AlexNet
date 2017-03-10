import theano
import theano.tensor as T
import numpy
import cPickle
import pickle
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

class ConvPoolLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data
                 inputShape,            # Shape of input = [batch size, channels, rows, cols]
                 filterShape,           # Shape of filter = [number of filters, channels, rows, cols]
                 borderMode = 'valid',  # Padding border with zeros
                 subsample = (1, 1),    # Filter stride
                 poolingShape = None,   # Shape of pooling (2, 2) default
                 W = None,
                 activation = T.tanh
                 ):
        # Set parameters
        self.Rng = rng
        self.Input = input
        self.InputShape = inputShape
        self.FilterShape = filterShape
        self.BorderMode = borderMode
        self.PoolingShape = poolingShape
        self.Subsample = subsample
        self.W = W
        self.Activation = activation

        self.createModel()

    def createModel(self):
        # Create shared parameters for filters
        if self.W is None:
            fanIn = numpy.prod(self.FilterShape[1:])
            fanOut = (self.FilterShape[0] * numpy.prod(self.FilterShape[2:]) // numpy.prod(self.Subsample))
            wBound = numpy.sqrt(6. / (fanIn + fanOut))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = self.FilterShape
                    ),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        convLayer = conv2d(
            input        = self.Input,
            input_shape  = self.InputShape,
            filters      = self.W,
            filter_shape = self.FilterShape,
            subsample    = self.Subsample,
            border_mode  = self.BorderMode

        )

        if self.PoolingShape is not None:
            poolLayer = pool_2d(
                input         = convLayer,
                ds            = self.PoolingShape,
                ignore_border = True
            )
            self.Output = poolLayer
        else:
            self.Output = convLayer

        self.Params = [self.W]

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow=True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self.Params]