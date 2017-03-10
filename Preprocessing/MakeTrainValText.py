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
from os import listdir
from os.path import abspath, join, isdir, isfile
from random import shuffle


def getListByIndices(list, indices):
    newList = [list[idx] for idx in indices]
    return newList


with open('../Utils/PathConfig.yaml', 'r') as f:
    paths = yaml.load(f)
TRAIN_IMG_DIR   = paths['TRAIN_IMG_DIR']
VALID_IMG_DIR   = paths['VALID_IMG_DIR']
TEST_IMG_DIR    = paths['TEST_IMG_DIR']

TRAIN_DATA_FILENAME         = paths['TRAIN_DATA_FILENAME']
VALID_DATA_FILENAME         = paths['VALID_DATA_FILENAME']
TEST_DATA_FILENAME          = paths['TEST_DATA_FILENAME']

SYNSETS         = paths['SYNSETS']

################################
#     Load synsets             #
################################
file = open(SYNSETS)
allSynsets = tuple(file)
allSynsets = [synset.strip().split(' ') for synset in allSynsets]
file.close()

def FindLabelFromSynset(folderName):
    for idx in range(allSynsets.__len__()):
        if allSynsets[idx][0] == folderName:
            return idx + 1


################################
#     Training dataset         #
################################
sortedTrainDirs = sorted([join(abspath(TRAIN_IMG_DIR), name) for name in listdir(TRAIN_IMG_DIR)
                          if isdir(join(TRAIN_IMG_DIR, name))])
allTrainFiles = []
allTrainFilesLabelNumber = []
allTrainFilesLabelText = []
for idx, trainDir in enumerate(sortedTrainDirs):
    labelName = trainDir.split('/')[-1]
    labelNum  = FindLabelFromSynset(labelName)
    sortedTrainFiles = sorted([join(abspath(trainDir), name) for name in listdir(trainDir)
                              if isfile(join(trainDir, name))])
    allTrainFiles.extend(sortedTrainFiles)
    for i in range(len(sortedTrainFiles)):
        allTrainFilesLabelNumber.append(labelNum)
        allTrainFilesLabelText.append(labelName)

# Random data before save
newIdx = list(range(len(allTrainFiles)))
shuffle(newIdx)
allTrainFiles = getListByIndices(allTrainFiles, newIdx)
allTrainFilesLabelNumber = getListByIndices(allTrainFilesLabelNumber, newIdx)
allTrainFilesLabelText = getListByIndices(allTrainFilesLabelText, newIdx)

# Save metadata to text
file = open(TRAIN_DATA_FILENAME, 'wb')
for idx in range(len(allTrainFiles)):
    file.write('%s %d %s \n' % (allTrainFiles[idx], allTrainFilesLabelNumber[idx], allTrainFilesLabelText[idx]))
file.close()

################################
#     Validation dataset       #
################################
sortedValidFiles = sorted([join(abspath(VALID_IMG_DIR), name) for name in listdir(VALID_IMG_DIR)
                          if isfile(join(VALID_IMG_DIR, name))])
allValidFiles = sortedValidFiles
file = open('ILSVRC2010_validation_ground_truth.txt')
allValidFilesLabelNumber = tuple(file)
allValidFilesLabelNumber = [int(validLabel.strip()) for validLabel in allValidFilesLabelNumber]
file.close()

# Save metadata to text
file = open(VALID_DATA_FILENAME, 'wb')
for idx in range(len(allValidFiles)):
    file.write('%s %d \n' % (allValidFiles[idx], allValidFilesLabelNumber[idx]))
file.close()

################################
#     Test dataset             #
################################
sortedTestFiles = sorted([ join(abspath(TEST_IMG_DIR), name) for name in listdir(TEST_IMG_DIR)
                        if isfile(join(TEST_IMG_DIR, name))])
allTestFiles = sortedTestFiles

# Save metadata to text
file = open(TEST_DATA_FILENAME, 'wb')
for idx in range(len(allTestFiles)):
    file.write('%s \n' % (allTestFiles[idx]))
file.close()



