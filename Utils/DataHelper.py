import cv2
import theano
import numpy
from random import randint

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

        self.EpochTrain = 0
        self.EpochValid = 0
        self.EpochTest  = 0

        self.loadTrainFile()
        self.loadValidFile()
        self.loadTestFile()

    def loadData(self, subData):
        data = []
        labels = []
        for sample in subData:
            imagePath = sample[0]
            label = sample[1]
            im = cv2.imread(imagePath)
            idx, idy = [randint(0, 255 - 227), randint(0, 255 - 227)]
            im = im[idx : idx + 227, idy : idy + 227, :]
            data.append(im)
            labels.append(label)
        data = numpy.asarray(data, dtype = theano.config.floatX)
        return [data, labels]

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
            self.NumTrainData = self.AllTrainData.__len__()

    def NextTrainBatch(self, batchSize):
        lastIndex = self.IndexTrain
        if (self.IndexTrain + batchSize >= self.AllTrainData.__len__()):
            listSample = self.AllTrainData[self.IndexTrain : self.AllTrainData.__len__()]
            listSample.extend(self.AllTrainData[0 : batchSize - (self.AllTrainData.__len__() - self.IndexTrain)])
            self.IndexTrain = batchSize - (self.AllTrainData.__len__() - self.IndexTrain)
        else:
            listSample = self.AllTrainData[self.IndexTrain : self.IndexTrain + batchSize]
            self.IndexTrain = (self.IndexTrain + batchSize) % self.AllTrainData.__len__()
        if lastIndex > self.IndexTrain:
            self.EpochTrain += 1
        return self.loadData(listSample)

    def loadValidFile(self):
        if self.ValidFilePath is not None:
            allSamples = self.loadDataFromFile(self.ValidFilePath)
            self.AllValidData = []
            for sample in allSamples:
                if (sample != ''):
                    splitSample = sample.strip().split(' ')
                    splitSample[1] = int(splitSample[1])
                    self.AllValidData.append(splitSample)
            self.NumValidData = self.AllValidData.__len__()

    def NextValidBatch(self, batchSize):
        lastIndex = self.IndexValid
        if (self.IndexValid + batchSize >= self.AllValidData.__len__()):
            listSample = self.AllValidData[self.IndexValid : self.AllValidData.__len__()]
            listSample.extend(self.AllValidData[0 : batchSize - (self.AllValidData.__len__() - self.IndexValid)])
            self.IndexValid = batchSize - (self.AllValidData.__len__() - self.IndexValid)
        else:
            listSample = self.AllValidData[self.IndexValid : self.IndexValid + batchSize]
            self.IndexValid = (self.IndexValid + batchSize) % self.AllValidData.__len__()
        if lastIndex > self.IndexValid:
            self.EpochValid += 1
        return self.loadData(listSample)

    def loadTestFile(self):
        if self.TestFilePath is not None:
            allSamples = self.loadDataFromFile(self.TestFilePath)
            self.AllTestData = []
            for sample in allSamples:
                if (sample != ''):
                    self.AllTestData.append(sample.split(' '))
            self.NumTestData = self.AllTestData.__len__()

    def NextTestBatch(self, batchSize):
        lastIndex = self.IndexTest
        if (self.IndexTest + batchSize >= self.AllTestData.__len__()):
            listSample = self.AllTestData[self.IndexTest : self.AllTestData.__len__()]
            listSample.extend(self.AllTestData[0 : batchSize - (self.AllTestData.__len__() - self.IndexTest)])
            self.IndexTest = batchSize - (self.AllTestData.__len__() - self.IndexTest)
        else:
            listSample = self.AllTestData[self.IndexTest : self.IndexTest + batchSize]
            self.IndexTest = (self.IndexTest + batchSize) % self.AllTestData.__len__()
        if lastIndex > self.IndexTest:
            self.EpochTest += 1
        return self.loadData(listSample)