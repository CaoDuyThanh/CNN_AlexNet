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
import os
import numpy

with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)

trainImgDir   = paths['TRAIN_IMG_DIR']
validImgDir   = paths['VALID_IMG_DIR']
trainFilename = paths['TRAIN_FILENAME']
validFilename = paths['VALID_FILENAME']

sortedTrainDirs = sorted([name for name in os.listdir(trainImgDir)
                          if os.path.isdir(os.path.join(trainImgDir, name))])


