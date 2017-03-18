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
import pickle
from pylab import *
from PIL import Image
from Utils.DataHelper import *

with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)

TRAIN_DATA_FILENAME         = paths['TRAIN_DATA_FILENAME']
VALID_DATA_FILENAME         = paths['VALID_DATA_FILENAME']
TEST_DATA_FILENAME          = paths['TEST_DATA_FILENAME']

################################
#     Load metadata            #
################################
dataHelper = DatasetHelper(
    trainFilePath = TRAIN_DATA_FILENAME,
    validFilePath = VALID_DATA_FILENAME,
    testFilePath = TEST_DATA_FILENAME
)
allTrainData = dataHelper.AllTrainData
allValidData = dataHelper.AllValidData
allTestData  = dataHelper.AllTestData

################################
#     Training dataset         #
################################
meanImage = numpy.zeros((256, 256, 3), dtype = 'float64')
counter = 0
for trainData in allTrainData:
    counter += 1
    if counter % 5000 == 0:
        print ('Process %d / %d images !' % (counter, allTrainData.__len__()))

    imagePath = trainData[0]
    image = Image.open(imagePath)
    meanImage += image
meanImage /= allTrainData.__len__()

file = open('mean_image.pkl', 'wb')
pickle.dump(meanImage, file, -1)
file.close()







