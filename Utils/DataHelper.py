import cv2
import theano
import numpy
import Queue
import time
from random import randint
from thread import start_new_thread

NUM_PRE_BATCH_LOAD = 100

def loadData(subData):
    data = numpy.zeros((subData.__len__(), 227, 227, 3), dtype = 'float32')
    labels = []
    for count, sample in enumerate(subData):
        imagePath = sample[0]
        label = sample[1]
        im = cv2.imread(imagePath)
        idx, idy = [randint(0, 255 - 227), randint(0, 255 - 227)]
        im = im[idx: idx + 227, idy: idy + 227, :]
        data[count] = im
        labels.append(label)
    data = data / 255.  # Normalize image to 0 - 1
    return [data, labels]

def NextBatch(allData, epoch, idx, batchSize):
    lastIndex = idx
    if (idx + batchSize >= allData.__len__()):
        listSample = allData[idx : allData.__len__()]
        listSample.extend(allData[0 : batchSize - (allData.__len__() - idx)])
        idx = batchSize - (allData.__len__() - idx)
    else:
        listSample = allData[idx : idx + batchSize]
        idx = (idx + batchSize) % allData.__len__()
    if lastIndex > idx:
        epoch += 1
    return [loadData(listSample), epoch, idx]

# -------------------------  THREADING  ---------------------------------------------------------------
def AutoLoadData(batches, allData, startEpoch, startIdx, batchSize):
    while (True):
        while (batches.qsize() < NUM_PRE_BATCH_LOAD):
            nextBatch, epoch, idx = NextBatch(allData, startEpoch, startIdx, batchSize)
            batches.put([nextBatch, epoch, idx])
            startEpoch = epoch
            startIdx   = idx
        time.sleep(1)


class DatasetHelper:
    def __init__(self,
                 trainFilePath = None,
                 validFilePath = None,
                 testFilePath  = None,
                 batchSize     = 64):
        self.TrainFilePath = trainFilePath
        self.ValidFilePath = validFilePath
        self.TestFilePath  = testFilePath
        self.BatchSize     = batchSize

        self.IndexTrain = 0
        self.IndexValid = 0
        self.IndexTest  = 0

        self.EpochTrain = 0
        self.EpochValid = 0
        self.EpochTest  = 0

        self.TrainBatches = Queue.Queue()
        self.ValidBatches = Queue.Queue()
        self.TestBatches  = Queue.Queue()

        self.loadTrainFile()
        self.loadValidFile()
        self.loadTestFile()

        # Start auto load data thread
        start_new_thread(AutoLoadData, (self.TrainBatches, self.AllTrainData, 0, 0, self.BatchSize))
        start_new_thread(AutoLoadData, (self.ValidBatches, self.AllValidData, 0, 0, self.BatchSize))
        start_new_thread(AutoLoadData, (self.TestBatches,  self.AllTestData , 0, 0, self.BatchSize))

    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
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

    def loadTestFile(self):
        if self.TestFilePath is not None:
            allSamples = self.loadDataFromFile(self.TestFilePath)
            self.AllTestData = []
            for sample in allSamples:
                if (sample != ''):
                    self.AllTestData.append(sample.split(' '))
            self.NumTestData = self.AllTestData.__len__()

    # -----------------------------------------------------------------------------------------------------
    def NextTrainBatch(self):
        while (True):
            if self.TrainBatches.qsize() > 0:
                break
            time.sleep(1)
        batchData = self.TrainBatches.get()
        batch, self.EpochTrain, idx = batchData
        return batch[0], batch[1], self.EpochTrain, idx

    def NextValidBatch(self):
        while (True):
            if self.ValidBatches.qsize() > 0:
                break
            time.sleep(1)
        batchData = self.ValidBatches.get()
        batch, self.EpochValid, idx = batchData
        return batch[0], batch[1], self.EpochValid, idx

    def NextTestBatch(self, batchSize):
        while (True):
            if self.TestBatches.qsize() > 0:
                break
            time.sleep(1)
        batchData = self.TestBatches.get()
        batch, self.EpochTest, idx = batchData
        return batch[0], batch[1], self.EpochTest, idx