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
                 b = None
                 ):
        # Set parameters
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation

        # Create shared parameters for hidden layer
        if W is None:
            """ We create random weights (uniform distribution) """
            # Create boundary for uniform generation
            wBound = numpy.sqrt(6.0 / (self.NumIn + self.NumOut))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(
                        low=-wBound,
                        high=wBound,
                        size=(self.NumIn, self.NumOut)
                    ),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            """ Or simply set weights from parameter """
            self.W = W

        if b is None:
            """ We create zeros bias """
            # Create bias
            self.b = theano.shared(
                numpy.zeros(
                    shape = (self.NumOut, ),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            """ Or simply set bias from parameter """
            self.b = b

    def Output(self):
        output = T.dot(self.Input, self.W) + self.b
        if self.Activation is None:
            return output
        else:
            return self.Activation(output)

    def Params(self):
        return [self.W, self.b]

