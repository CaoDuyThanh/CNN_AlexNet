'''
Generating caffe style train and validation label txt files from dataset
Caffe style
    Training set
        Path        Label
        Path        Label
        Path        Label
        ...............
'''

import yaml
import numpy
from PIL import Image

with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)

TRAIN_DATA_FILENAME         = paths['TRAIN_DATA_FILENAME']

################################
#     Training dataset         #
################################
file = open(TRAIN_DATA_FILENAME)
allSamples = tuple(file)
file.close()

numSamples = 0
allTrainData = []
for sample in allSamples:
    if (sample != ''):
        numSamples += 1
        allTrainData.append(sample.split(' '))

meanImage = numpy.zeros((3, 224, 224), dtype = 'float64')
for trainData in allTrainData:
    imagePath = trainData[0]
    image = Image.open(imagePath)
    image = Image.fromarray(numpy.rollaxis(image, 0, 3))
    meanImage += image
meanImage /= numSamples





