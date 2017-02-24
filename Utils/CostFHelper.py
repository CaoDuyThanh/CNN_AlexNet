import theano
import theano.tensor as T
import numpy

""" L1 - Regularization """
def L1(W):
    return abs(W).sum()

""" L2 - Regularization """# return -T.mean(T.neq(T.argmax(output), T.argmax(y)))
def L2(W):
    return abs(W ** 2).sum()

""" Cross entropy """
def CrossEntropy(output, y):
    # return T.mean(T.sum(T.nnet.binary_crossentropy(output, y), 1))
    return -T.mean(T.log(output)[T.arange(y.shape[0]), y])


""" Error """
def Error(output, y):
    # return -T.mean(T.neq(T.argmax(output), T.argmax(y)))
    return T.mean(T.neq(T.argmax(output, 1), y))