import theano
import theano.tensor as T
import numpy
import os
import gzip
import pickle

class DatasetHelper:
    def __init__(self,
                 trainFilePath = None,
                 validFilePath = None,
                 testFilePath = None):
        self.TrainFilePath = trainFilePath
        self.ValidFilePath = validFilePath
        self.TestFilePath = testFilePath

        self.IndexTrain = 0
        self.IndexValid = 0
        self.IndexTest  = 0

        self.loadTrainFile()
        self.loadValidFile()
        self.loadTestFile()

    def loadDataFromFile(self, path):
        file = open(path)
        allData = tuple(file)
        file.close()
        return allData

    def loadTrainFile(self):
        if self.TrainFilePath is not None:
            allSamples = self.loadDataFromFile(self.TrainFilePath)
            self.AllTrainData = []

            for sample in allSamples:
                if (sample != ''):
                    splitSample = sample.strip().split(' ')
                    splitSample[1] = int(splitSample[1])
                    self.AllTrainData.append(splitSample)

    def NextTrainBatch(self, batchSize):
        if (self.IndexTrain + batchSize >= self.AllTrainData.__len__()):
            listSample = self.AllTrainData[self.IndexTrain : self.AllTrainData.__len__()]
            listSample.append(self.AllTrainData[0 : batchSize - (self.AllTrainData.__len__() - self.IndexTrain)])
            self.IndexTrain = batchSize - (self.AllTrainData.__len__() - self.IndexTrain)
        else:
            listSample = self.AllTrainData[self.IndexTrain : self.IndexTrain + batchSize]
            self.IndexTrain = (self.IndexTrain + batchSize) % self.AllTrainData.__len__()

    def loadValidFile(self):
        if self.ValidFilePath is not None:
            allSamples = self.loadDataFromFile(self.ValidFilePath)
            self.AllValidData = []

            for sample in allSamples:
                if (sample != ''):
                    splitSample = sample.strip().split(' ')
                    splitSample[1] = int(splitSample[1])
                    self.AllValidData.append(splitSample)

    def loadTestFile(self):
        if self.TestFilePath is not None:
            allSamples = self.loadDataFromFile(self.TestFilePath)
            self.AllTestData = []

            for sample in allSamples:
                if (sample != ''):
                    self.AllTestData.append(sample.split(' '))
