import yaml
import numpy
import PIL
from PIL import Image
from Utils.DataHelper import *

with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)

with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)

TRAIN_DATA_FILENAME         = paths['TRAIN_DATA_FILENAME']
VALID_DATA_FILENAME         = paths['VALID_DATA_FILENAME']
TEST_DATA_FILENAME          = paths['TEST_DATA_FILENAME']

def toRGB(imageIn):
    width, height = imageIn.size
    image = numpy.zeros((width, height, 3), dtype = 'uint8')
    image[:, : , 0] = imageIn
    image[:, : , 1] = imageIn
    image[:, : , 2] = imageIn
    image = Image.fromarray(image)
    return image

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
print ('Training dataset preprocessing ! CROP IMAGE AND RESIZE TO (256, 256)')
counter = 0
for trainData in allTrainData:
    counter += 1
    if counter % 500 == 0:
        print ('Process %d / %d images !' % (counter, allTrainData.__len__()))

    imagePath = trainData[0]
    image = Image.open(imagePath)
    width = image.size[0]
    height = image.size[1]
    scale = 256.0 / min(height, width)
    width = int(width * scale)
    height = int(height * scale)
    image = image.resize((width, height), PIL.Image.BILINEAR)
    width = image.size[0]
    height = image.size[1]
    if (width < height):
        image = image.crop((
            0,
            int(height / 2 - 128),
            256,
            int(height / 2 - 128) + 256,
        ))
    else:
        image = image.crop((
            int(width / 2 - 128),
            0,
            int(width / 2 - 128) + 256,
            256
        ))
    if image.mode == 'L':
        image = toRGB(image)
    image.save(imagePath)

################################
#     Validation dataset       #
################################
print ('Validation dataset preprocessing ! CROP IMAGE AND RESIZE TO (256, 256)')
counter = 0
for validData in allValidData:
    counter += 1
    if counter % 500 == 0:
        print ('Process %d / %d images !' % (counter, allValidData.__len__()))

    imagePath = validData[0]
    image = Image.open(imagePath)
    width = image.size[0]
    height = image.size[1]
    scale = 256.0 / min(height, width)
    width = int(width * scale)
    height = int(height * scale)
    image = image.resize((width, height), PIL.Image.BILINEAR)
    width = image.size[0]
    height = image.size[1]
    if (width < height):
        image = image.crop((
            0,
            int(height / 2 - 128),
            256,
            int(height / 2 - 128) + 256,
        ))
    else:
        image = image.crop((
            int(width / 2 - 128),
            0,
            int(width / 2 - 128) + 256,
            256
        ))
    image.save(imagePath)

################################
#     Testing dataset          #
################################
print ('Testing dataset preprocessing ! CROP IMAGE AND RESIZE TO (256, 256)')
counter = 0
for testData in allTestData:
    counter += 1
    if counter % 500 == 0:
        print ('Process %d / %d images !' % (counter, allTestData.__len__()))

    imagePath = testData[0]
    image = Image.open(imagePath)
    width = image.size[0]
    height = image.size[1]
    scale = 256.0 / min(height, width)
    width = int(width * scale)
    height = int(height * scale)
    image = image.resize((width, height), PIL.Image.BILINEAR)
    width = image.size[0]
    height = image.size[1]
    if (width < height):
        image = image.crop((
            0,
            int(height / 2 - 128),
            256,
            int(height / 2 - 128) + 256,
        ))
    else:
        image = image.crop((
            int(width / 2 - 128),
            0,
            int(width / 2 - 128) + 256,
            256
        ))
    image.save(imagePath)





