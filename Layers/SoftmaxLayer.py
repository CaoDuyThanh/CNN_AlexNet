from theano.tensor.nnet.nnet import softmax

class SoftmaxLayer:
    def __init__(self,
                 input):
        self.Input = input

    def Output(self):
        return softmax(self.Input)