import theano
import numpy
import theano.tensor as T

class HiddenLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data input
                 numIn,                 # Number neurons of input
                 numOut,                # Number reurons out of layer
                 activation = T.tanh,   # Activation function
                 W = None,
                 b = None):
        # Set parameters
        self.Rng = rng
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation
        self.W = W
        self.b = b

        self.createModel()

    def createModel(self):
        # Create shared parameters for hidden layer
        if self.W is None:
            """ We create random weights (uniform distribution) """
            # Create boundary for uniform generation
            wBound = numpy.sqrt(6.0 / (self.NumIn + self.NumOut))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumIn, self.NumOut)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.b is None:
            """ We create zeros bias """
            # Create bias
            self.b = theano.shared(
                numpy.zeros(
                    shape = (self.NumOut, ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        self.Params = [self.W, self.b]
        self.Output = self.Activation(T.dot(self.Input, self.W) + self.b)
