from __future__ import print_function
import yaml
import timeit
import os
import Utils.CostFHelper as CostFHelper
from Utils.DataHelper import *
from Layers.HiddenLayer import *
from Layers.ConvPoolLayer import *

# DATASET INFO
with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)
TRAIN_DATA_FILENAME = paths['TRAIN_DATA_FILENAME']
VALID_DATA_FILENAME = paths['VALID_DATA_FILENAME']
TEST_DATA_FILENAME  = paths['TEST_DATA_FILENAME']

# SAVE SETTINGS
SAVE_MODEL          = paths['SAVE_MODEL']

# OTHER SETTINGS
VALIDATION_FREQUENCY  = 5000
VISUALIZE_FREQUENCY   = 10

# TRAINING PARAMETERS
NUM_ITERATION = 700000
LEARNING_RATE = 0.01
BATCH_SIZE    = 64

# ADAM PARAMETERS
BETA1 = 0.9
BETA2 = 0.999
DELTA = 0.000001

# EARLY STOPPING PARAMETERS
PATIENCE              = 1000
PATIENCE_INCREASE     = 2
IMPROVEMENT_THRESHOLD = 0.995

# GLOBAL VARIABLES
Dataset   = None
MeanImage = None

def prepareDataset():
    global Dataset
    Dataset = DatasetHelper(
        trainFilePath = TRAIN_DATA_FILENAME,
        validFilePath = VALID_DATA_FILENAME,
        testFilePath  = TEST_DATA_FILENAME
    )

def loadMeanImage():
    global MeanImage
    file = open('mean_image.pkl')
    meanImage = cPickle.load(file)
    meanImage = numpy.asarray(meanImage, dtype = theano.config.floatX)
    meanImage = meanImage[15 : 15 + 227, 15 : 15 + 227, :]
    meanImage = theano.shared(meanImage, borrow = True)
    MeanImage = meanImage.dimshuffle('x', 2, 0, 1)
    MeanImage = MeanImage / 255.0   # Normalize image to 0 - 1
    file.close()

def evaluateAlexNet():
    ####################################
    #       Create model               #
    ####################################
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
    LearningRate = T.fscalar('LearningRate')
    X = T.tensor4('X')
    Y = T.ivector('Y')

    X4D = X.reshape((BATCH_SIZE, 227, 227, 3))
    X4DC = X4D.dimshuffle(0, 3, 1, 2) - MeanImage
    # Convolution & pooling layer 0
    convPoolLayer0  = ConvPoolLayer(
        rng         = rng,
        input       = X4DC,
        inputShape  = (BATCH_SIZE, 3, 227, 227),
        filterShape = (96, 3, 11, 11),
        subsample   = (4, 4),
        poolingShape = (2, 2),
        activation  = T.nnet.relu
    )
    convPoolLayer0Params = convPoolLayer0.Params
    convPoolLayer0Output = convPoolLayer0.Output

    # Convolution & pooling layer 1
    convPoolLayer1 = ConvPoolLayer(
        rng         = rng,
        input       = convPoolLayer0Output,
        inputShape  = (BATCH_SIZE, 96, 27, 27),
        filterShape = (256, 96, 5, 5),
        borderMode  = 2,
        poolingShape = (2, 2),
        activation  = T.nnet.relu
    )
    convPoolLayer1Params = convPoolLayer1.Params
    convPoolLayer1Output = convPoolLayer1.Output

    # Convolution & pooling layer 2
    convPoolLayer2 = ConvPoolLayer(
        rng         = rng,
        input       = convPoolLayer1Output,
        inputShape  = (BATCH_SIZE, 256, 13, 13),
        filterShape = (384, 256, 3, 3),
        borderMode  = 1,
        activation  = T.nnet.relu
    )
    convPoolLayer2Params = convPoolLayer2.Params
    convPoolLayer2Output = convPoolLayer2.Output

    # Convolution & pooling layer 3
    convPoolLayer3 = ConvPoolLayer(
        rng         = rng,
        input       = convPoolLayer2Output,
        inputShape  = (BATCH_SIZE, 384, 13, 13),
        filterShape = (384, 384, 3, 3),
        borderMode  = 1,
        activation  = T.nnet.relu
    )
    convPoolLayer3Params = convPoolLayer3.Params
    convPoolLayer3Output = convPoolLayer3.Output

    # Convolution & pooling layer 4
    convPoolLayer4 = ConvPoolLayer(
        rng         = rng,
        input       = convPoolLayer3Output,
        inputShape  = (BATCH_SIZE, 384, 13, 13),
        filterShape = (256, 384, 3, 3),
        borderMode  = 1,
        poolingShape = (2, 2),
        activation  = T.nnet.relu
    )
    convPoolLayer4Params = convPoolLayer4.Params
    convPoolLayer4Output = convPoolLayer4.Output
    convPoolLayer4OutputRes = convPoolLayer4Output.reshape((BATCH_SIZE, 256 * 6 * 6))

    # Hidden layer 0
    hidLayer0 = HiddenLayer(
        rng        = rng,
        input      = convPoolLayer4OutputRes,
        numIn      = 256 * 6 * 6,
        numOut     = 4096,
        activation = T.nnet.relu
    )
    hidLayer0Params = hidLayer0.Params
    hidLayer0Output = hidLayer0.Output

    # Hidden layer 1
    hidLayer1 = HiddenLayer(
        rng        = rng,
        input      = hidLayer0Output,
        numIn      = 4096,
        numOut     = 4096,
        activation = T.nnet.relu
    )
    hidLayer1Params = hidLayer1.Params
    hidLayer1Output = hidLayer1.Output

    # Hidden layer 2
    hidLayer2 = HiddenLayer(
        rng        = rng,
        input      = hidLayer1Output,
        numIn      = 4096,
        numOut     = 1000,
        activation = T.nnet.softmax
    )
    hidLayer2Params = hidLayer2.Params
    hidLayer2Output = hidLayer2.Output

    Layers = [convPoolLayer0,
              convPoolLayer1,
              convPoolLayer2,
              convPoolLayer3,
              convPoolLayer4,
              hidLayer0,
              hidLayer1,
              hidLayer2]
    Output = hidLayer2Output

    ####################################
    #       Create functions           #
    ####################################
    # Cost | Error function
    costFunc = CostFHelper.CrossEntropy(Output, Y)
    errFunc = CostFHelper.Error(Output, Y)

    # List of params from model
    params = hidLayer2Params + \
             hidLayer1Params + \
             hidLayer0Params + \
             convPoolLayer4Params + \
             convPoolLayer3Params + \
             convPoolLayer2Params + \
             convPoolLayer1Params + \
             convPoolLayer0Params

    # Define gradient
    grads = T.grad(costFunc, params)
    # Updates function
    updates = []
    for (param, grad) in zip(params, grads):
        mt = theano.shared(param.get_value() * 0., broadcastable = param.broadcastable)
        vt = theano.shared(param.get_value() * 0., broadcastable = param.broadcastable)

        newMt = BETA1 * mt + (1 - BETA1) * grad
        newVt = BETA2 * vt + (1 - BETA2) * T.sqr(grad)

        tempMt = newMt / (1 - BETA1)
        tempVt = newVt / (1 - BETA2)

        step = - LearningRate * tempMt / (T.sqrt(tempVt) + DELTA)
        updates.append((mt, newMt))
        updates.append((vt, newVt))
        updates.append((param, param + step))

    # Train model
    trainFunc = theano.function(
        inputs  = [X, Y, LearningRate],
        outputs = costFunc,
        updates = updates
    )

    testFunc = theano.function(
        inputs  = [X, Y],
        outputs = errFunc
    )

    ####################################
    #       Training models            #
    ####################################
    # Load old model
    if os.path.isfile(SAVE_MODEL):
        file = open(SAVE_MODEL)
        [layer.LoadModel(file) for layer in Layers]
        file.close()

    # Learning rate
    dynamicLearningRate = LEARNING_RATE

    # Early stopping
    patience = PATIENCE

    # Best model | error
    bestError = 10

    startTime = timeit.default_timer()
    timeVisualize = timeit.default_timer()
    for iter in range(NUM_ITERATION):
        if (iter % VALIDATION_FREQUENCY == 0):
            print('     Validate model....')
            epochValid = Dataset.EpochValid
            err = 0; numValidSamples = 0; validIter = 0

            validStart = timeit.default_timer()
            while epochValid == Dataset.EpochValid:
                validIter += 1
                [subData, labels] = Dataset.NextValidBatch(BATCH_SIZE)
                err += testFunc(subData, labels)
                numValidSamples += BATCH_SIZE
                if validIter % VISUALIZE_FREQUENCY == 0:
                    print ('     Iterations = %d, NumValidSamples = %d' % (validIter, numValidSamples))
            err /= validIter
            validEnd = timeit.default_timer()
            print('     Validation complete! Time validation = %f mins. Error = %f (previous best error = %f)' % ((validEnd - validStart) / 60., err, bestError))

            if (err < bestError):
                if (err < bestError * IMPROVEMENT_THRESHOLD):
                    patience = max(patience, iter * PATIENCE_INCREASE)
                bestError = err

                # Save model
                file = open(SAVE_MODEL, 'wb')
                [layer.SaveModel(file) for layer in Layers]
                file.close()
                print('Save model!')

        # Load data
        [subData, labels] = Dataset.NextTrainBatch(BATCH_SIZE)
        epochTrain = Dataset.EpochTrain

        # Train model
        cost = trainFunc(subData, labels, dynamicLearningRate)

        if (iter % VISUALIZE_FREQUENCY == 0):
            oldTimeVisualize = timeVisualize
            timeVisualize = timeit.default_timer()
            print('Epoch = %d, iteration = %d, cost = %f. Time remain = %f' % (epochTrain, iter, cost, (
            timeVisualize - oldTimeVisualize) / 60. * (NUM_ITERATION - iter) / VISUALIZE_FREQUENCY))

                    # if (patience < iter):
        #     print ('Early stopping !')
        #     print('Epoch = %d, iteration = %d, cost = %f' % (epochTrain, iter, cost))
        #     break

    endTime = timeit.default_timer()


if __name__ == "__main__":
    prepareDataset()
    loadMeanImage()
    evaluateAlexNet()