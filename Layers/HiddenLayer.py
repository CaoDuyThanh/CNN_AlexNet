import theano
import numpy
import pickle
import cPickle
import theano.tensor as T

class HiddenLayer:
    def __init__(self,
                 rng,                   # Random seed
                 theanoRng,             # Random theano seed
                 input,                 # Data input
                 numIn,                 # Number neurons of input
                 numOut,                # Number reurons out of layer
                 activation = T.tanh,   # Activation function
                 dropout = None,         # Dropout parameter
                 W = None,
                 b = None):
        # Set parameters
        self.Rng = rng
        self.TheanoRng = theanoRng
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation
        self.Dropout = dropout
        self.W = W
        self.b = b

        self.createModel()

    def dropout(self, input, corruptionLevel):
        return self.TheanoRng.binomial(size=input.shape, n=1,
                                       p = 1 - corruptionLevel,
                                       dtype=theano.config.floatX) * input

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

        if self.Dropout is not None:
            self.Output = self.dropout(self.Output, self.Dropout)


    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow=True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self.Params]