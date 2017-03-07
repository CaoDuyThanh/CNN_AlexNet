import theano
import theano.tensor as T
import numpy
import os
import gzip
import pickle

class DatasetHelper:
    def __init__(self,
                 trainFilePath,
                 validFilePath,
                 testFilePath):
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
        allSamples = self.loadDataFromFile(self.TrainFilePath)
        self.AllTrainData = []

        for sample in allSamples:
            if (sample != ''):
                self.AllTrainData.append(sample.split(' '))

    def NextTrainBatch(self, batchSize):
        if (self.IndexTrain + batchSize >= self.AllTrainData.__len__()):
            listSample = self.AllTrainData[self.IndexTrain : self.AllTrainData.__len__()]
            listSample.append(self.AllTrainData[0 : batchSize - (self.AllTrainData.__len__() - self.IndexTrain)])
            self.IndexTrain = batchSize - (self.AllTrainData.__len__() - self.IndexTrain)
        else:
            listSample = self.AllTrainData[self.IndexTrain : self.IndexTrain + batchSize]
            self.IndexTrain = (self.IndexTrain + batchSize) % self.AllTrainData.__len__()

    def loadValidFile(self):
        allSamples = self.loadDataFromFile(self.ValidFilePath)
        self.AllValidData = []

        for sample in allSamples:
            if (sample != ''):
                self.AllValidData.append(sample.split(' '))

    def loadTestFile(self):
        allSamples = self.loadDataFromFile(self.TestFilePath)
        self.AllTestData = []

        for sample in allSamples:
            if (sample != ''):
                self.AllTestData.append(sample.split(' '))
