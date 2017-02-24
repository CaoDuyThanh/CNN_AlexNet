from __future__ import print_function
import timeit
import sys
import Utils.DataHelper as DataHelper
import Utils.CostFHelper as CostFHelper
from Layers.HiddenLayer import *
from Layers.SoftmaxLayer import *
from Layers.ConvPoolLayer import *

# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.005
NUM_EPOCH = 1000
BATCH_SIZE = 20
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 500

def evaluateAlexNet():
    # Load datasets from local disk or download from the internet


    nTrainBatchs = 0
    nValidBatchs = 0
    nTestBatchs  = 0

    # Create model
    '''
    MODEL ARCHITECTURE
    INPUT     ->    Convolution      ->        Dropout
    (32x32)        (6, 1, 5, 5)              (6, 14, 14)
              ->    Convolution      ->        Dropout
                   (16, 6, 5, 5)             (16, 5, 5)
              ->    Hidden layer
                    (120 neurons)
              ->    Hidden layer
                    (84 neurons)
              ->    Output layer (Softmax)
                    (10 neurons)
    '''
    # Create random state
    rng = numpy.random.RandomState(12345)

    # Create shared variable for input
    Index = T.lscalar('Index')
    X = T.matrix('X')
    Y = T.ivector('Y')

    X4D = X.reshape((BATCH_SIZE, 3, 224, 224))
    # Convolution & pooling layer 0
    convPoolLayer0 = ConvPoolLayer(
        rng = rng,
        input = X4D,
        inputShape = (BATCH_SIZE, 3, 224, 224),
        filterShape = (96, 3, 11, 11),
        subsample = (4, 4)
    )
    convPoolLayer0Output = convPoolLayer0.Output()
    convPoolLayer0Params = convPoolLayer0.Params()

    # Convolution & pooling layer 1
    convPoolLayer1 = ConvPoolLayer(
        rng = rng,
        input = convPoolLayer0Output,
        inputShape = (BATCH_SIZE, 96, 55, 55),
        filterShape=(256, 96, 5, 5)
    )
    convPoolLayer1Output = convPoolLayer1.Output()
    convPoolLayer1Params = convPoolLayer1.Params()

    # Convolution & pooling layer 2
    convPoolLayer2 = ConvPoolLayer(
        rng=rng,
        input=convPoolLayer1Output,
        inputShape=(BATCH_SIZE, 256, 27, 27),
        filterShape=(384, 256, 3, 3)
    )
    convPoolLayer2Output = convPoolLayer2.Output()
    convPoolLayer2Params = convPoolLayer2.Params()

    # Convolution & pooling layer 3
    convPoolLayer3 = ConvPoolLayer(
        rng=rng,
        input=convPoolLayer2Output,
        inputShape=(BATCH_SIZE, 384, 13, 13),
        filterShape=(384, 384, 3, 3)
    )
    convPoolLayer3Output = convPoolLayer3.Output()
    convPoolLayer3Params = convPoolLayer3.Params()

    # Convolution & pooling layer 4
    convPoolLayer4 = ConvPoolLayer(
        rng=rng,
        input=convPoolLayer3Output,
        inputShape=(BATCH_SIZE, 384, 13, 13),
        filterShape=(256, 384, 3, 3)
    )
    convPoolLayer4Output = convPoolLayer4.Output()
    convPoolLayer4Params = convPoolLayer4.Params()
    convPoolLayer4OutputRes = convPoolLayer4Output.reshape(BATCH_SIZE, 256 * 13 * 13)

    # Hidden layer 0
    hidLayer0 = HiddenLayer(
        rng = rng,
        input = convPoolLayer4OutputRes,
        numIn = 256 * 13 * 13,
        numOut = 4096,
        activation = T.tanh
    )
    hidLayer0Output = hidLayer0.Output()
    hidLayer0Params = hidLayer0.Params()

    # Hidden layer 1
    hidLayer1 = HiddenLayer(
        rng = rng,
        input = hidLayer0Output,
        numIn = 4096,
        numOut = 4096,
        activation = T.tanh
    )
    hidLayer1Output = hidLayer1.Output()
    hidLayer1Params = hidLayer1.Params()

    # Hidden layer 2
    hidLayer2 = HiddenLayer(
        rng = rng,
        input = hidLayer1Output,
        numIn = 4096,
        numOut = 1000,
        activation = T.tanh
    )
    hidLayer2Output = hidLayer2.Output()
    hidLayer2Params = hidLayer2.Params()

    # Softmax layer
    softmaxLayer0 = SoftmaxLayer(
        input=hidLayer2Output
    )
    softmaxLayer0Output = softmaxLayer0.Output()

    # List of params from model
    params = hidLayer2Params + \
             hidLayer1Params + \
             hidLayer0Params + \
             convPoolLayer4Params + \
             convPoolLayer3Params + \
             convPoolLayer2Params + \
             convPoolLayer1Params + \
             convPoolLayer0Params

    # Evaluate model - using early stopping
    # Define cost function = Regularization + Cross entropy of softmax
    costTrain = CostFHelper.CrossEntropy(softmaxLayer0Output, Y)

    # Define gradient
    grads = T.grad(costTrain, params)

    # Updates function
    updates = [
        (param, param - LEARNING_RATE * grad)
        for (param, grad) in zip(params, grads)
        ]

    # Train model
    trainModel = theano.function(
        inputs=[Index],
        outputs=costTrain,
        updates=updates,
        givens={
            X: trainSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: trainSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    # error = CostFHelper.Error(softmaxLayer0Output, Y)
    # # Valid model
    # validModel = theano.function(
    #     inputs=[Index],
    #     outputs=error,
    #     givens={
    #         X: validSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
    #         Y: validSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
    #     }
    # )
    #
    # # Test model
    # testModel = theano.function(
    #     inputs=[Index],
    #     outputs=error,
    #     givens={
    #         X: testSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
    #         Y: testSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
    #     }
    # )
    #
    # doneLooping = False
    # iter = 0
    # patience = PATIENCE
    # best_error = 1
    # best_iter = 0
    # start_time = timeit.default_timer()
    # epoch = 0
    # while (epoch < NUM_EPOCH) and (not doneLooping):
    #     epoch = epoch + 1
    #     for indexBatch in range(nTrainBatchs):
    #         iter = (epoch - 1) * nTrainBatchs + indexBatch
    #         cost = trainModel(indexBatch)
    #
    #         if iter % VALIDATION_FREQUENCY == 0:
    #             print('Validate model....')
    #             err = 0;
    #             for indexValidBatch in range(nValidBatchs):
    #                 err += validModel(indexValidBatch)
    #             err /= nValidBatchs
    #             print('Error = ', err)
    #
    #             if (err < best_error):
    #                 if (err < best_error * IMPROVEMENT_THRESHOLD):
    #                     patience = max(patience, iter * PATIENCE_INCREASE)
    #
    #                 best_iter = iter
    #                 best_error = err
    #
    #                 # Test on test set
    #                 test_losses = [testModel(i) for i in range(nTestBatchs)]
    #                 test_score = numpy.mean(test_losses)
    #
    #     if (patience < iter):
    #         doneLooping = True
    #         break
    #
    # end_time = timeit.default_timer()
    # print(('Optimization complete. Best validation score of %f %% '
    #        'obtained at iteration %i, with test performance %f %%') %
    #       (best_error * 100., best_iter + 1, test_score * 100.))
    # print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == "__main__":
    evaluateAlexNet()