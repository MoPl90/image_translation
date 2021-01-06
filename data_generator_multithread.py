from tensorflow.keras import backend as K

import numpy as np
import tensorflow.keras as keras
import pydicom
import glob
from PIL import Image
from augmentation import augmentor
import datetime
import time
import os
import matplotlib.pyplot as plt
import nibabel as nib
from data_generator import *

import threading
import queue

#--------------------
#SETTINGS
#--------------------
NUM_THREADS = 3
MAX_LENGTH_QUEUE = 3 # +NUM_THREADS samples will be loaded and wait to put if sampleQueue is full
#--------------------

class DataGeneratorMultiThread(keras.utils.Sequence):
    def __init__(self, 
                 epochs, 
                 list_IDs, 
                 imagePath, 
                 labelPath, 
                 normalization_args, 
                 augment = 0, 
                 augmentation_args = {}, 
                 preload_data = False, 
                 imageType = '.dcm', 
                 labelType = '.dcm', 
                 batch_size=32, 
                 dim=(32, 32, 32), 
                 crop_parameters=[0,10,0,10,0,10], 
                 n_channels=1, 
                 n_classes=10, 
                 shuffle=True, 
                 variableTypeX = 'float32', 
                 variableTypeY = 'float32', 
                 **kwargs):

        #params for sample identification & ordering
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        
        #create augmentor param set
        augmentation_params = {
                    'rotationAngle_Xaxis': augmentation_args['rotationRangeXAxis'],
                    'rotationAngle_Yaxis': augmentation_args['rotationRangeYAxis'],
                    'rotationAngle_Zaxis': augmentation_args['rotationRangeZAxis'],
                    'shift_Xaxis': augmentation_args['shiftXAxisRange'],
                    'shift_Yaxis': augmentation_args['shiftYAxisRange'],
                    'shift_Zaxis': augmentation_args['shiftZAxisRange'],
                    'stretch_Xaxis': augmentation_args['stretchFactorXAxisRange'],
                    'stretch_Yaxis': augmentation_args['stretchFactorYAxisRange'],
                    'stretch_Zaxis': augmentation_args['stretchFactorZAxisRange'],
                    'shear_NormalXAxis': augmentation_args['shear_NormalXAxisRange'],
                    'shear_NormalYAxis': augmentation_args['shear_NormalYAxisRange'],
                    'shear_NormalZAxis': augmentation_args['shear_NormalZAxisRange'],
                    'zoom': augmentation_args['zoomRange'],
                    'flip': augmentation_args['flip'],
                    }
        
        #create data loader instance (where samples will be prepared for training)
        self.dataLoader = DataLoader(epochs, list_IDs, imagePath, labelPath, normalization_args, augment, augmentation_params, augmentation_args['maxNumberOfTransformation'], imageType, labelType, dim, crop_parameters, n_channels, n_classes, variableTypeX, variableTypeY, batch_size, shuffle)
        
        #set index order for first epoch
        self.epoch_count = 0
        self.on_epoch_end()


    def on_epoch_end(self):
        self.dataLoader.end_of_epoch(self.epoch_count)
        self.n = 0
        self.epoch_count += 1
    
    
    def terminate(self):
        self.dataLoader.terminate()
    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        #get 'batch_size' samples from DataLoader
        X, Y = [], []
        for i in range(self.batch_size):
            X_temp, Y_temp = self.dataLoader.getLoadedSample()
            X.append(X_temp)
            Y.append(Y_temp)
        
        #concatenate to batch
        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)

        return X, Y

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return data

#-------------------------------------------------------------------------
# DataLoader class handles loading and preprocessing of data for training 
#-------------------------------------------------------------------------
class DataLoader():
    def __init__(self, epochs, list_IDs, imagePath, labelPath, normalization_args, augment, augmentation_params, numMaxTransforms, imageType, labelType, dim, crop_parameters, n_channels, n_classes, variableTypeX, variableTypeY, batch_size, shuffle):
        
        #general settings
        self.maxEpochs = epochs
        self.list_IDs = list_IDs
        self.imagePath = imagePath
        self.labelPath = labelPath
        self.normalization_args = normalization_args
        self.augment = augment
        self.augmentation_params = augmentation_params
        self.numMaxTransforms = numMaxTransforms
        self.imageType = imageType
        self.labelType = labelType
        self.dim = dim
        self.crop_parameters = crop_parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.variableTypeX = variableTypeX
        self.variableTypeY = variableTypeY
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        #define queue for indexes & loaded and preprocessed samples
        self.indexQueue = queue.Queue()
        self.sampleQueue = queue.Queue()
        self.indexQLock = threading.Lock()
        self.sampleQLock = threading.Lock()
        
        #define queue full semaphore and stop-threads event
        self.queueSemaphore = threading.Semaphore(value=MAX_LENGTH_QUEUE)
        self.stopFlag = threading.Event()
        self.stopFlag.clear()
        
        #define number of samples to be processed per epoch
        self.perEpochSamples = len(self.list_IDs) + self.batch_size
        self.epochSamples = self.perEpochSamples
        
        #create new threads for image loading and preprocessing
        self.threads = []
        for thID in range(NUM_THREADS):
            self.threads.append(LoadingThread(thID, self.list_IDs, self.imagePath, self.labelPath, self.normalization_args, self.augment, self.augmentation_params, self.numMaxTransforms,
                                             self.imageType, self.labelType, self.dim, self.crop_parameters, self.n_channels, self.n_classes, self.variableTypeX, self.variableTypeY,
                                             self.indexQueue, self.sampleQueue, self.indexQLock, self.sampleQLock, self.queueSemaphore, self.stopFlag))
        
        #start threads
        for thread in self.threads:
            thread.start()
    
    
    def terminate(self):
        #set stop flag
        self.stopFlag.set()
        
        #remove remaining samples
        while not self.sampleQueue.empty():
            self.sampleQLock.acquire()
            self.sampleQueue.get()
            self.queueSemaphore.release()
            self.epochSamples += 1
            self.sampleQLock.release()
        
        #wait for all threads to finish
        for thd in self.threads:
            thd.join()
        self.threads.clear()
        
        return
    
    
    def updateIndexes(self):
        #updates indexes after each epoch
        indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
            
        #fill index queue with new index ordering
        self.indexQLock.acquire()
        for idx in indexes:
            self.indexQueue.put(idx)
        
        #add indexes of first batch 2 times (because __getitem__ is (always?) accessed 2 times with index=0)
        for i in range(self.batch_size):
            self.indexQueue.put(indexes[i])
        
        self.indexQLock.release()
        
    
    def end_of_epoch(self, epoch):     
        #terminate if end of last epoch
        if epoch == self.maxEpochs:
            #set stop flag and wait for all threads to finish
            self.stopFlag.set()
            for thd in self.threads:
                thd.join()
            self.threads.clear()
            return
        
        #remove remaining samples from epoch (if any)
        self.sampleQLock.acquire()
        while not self.epochSamples == self.perEpochSamples:
            self.sampleQueue.get()
            self.queueSemaphore.release()
            self.epochSamples += 1
        self.sampleQLock.release()
        
        #update indexes to continue loading
        #indexes are added one epoch in prior
        self.epochSamples = 0
        if epoch == 0:
            self.updateIndexes()
            self.updateIndexes()
        elif epoch < (self.maxEpochs - 1):
            self.updateIndexes()


    def getLoadedSample(self):
        #wait for sample
        while self.sampleQueue.empty():
            pass
        
        #get sample
        self.sampleQLock.acquire()
        X, Y = self.sampleQueue.get()
        self.queueSemaphore.release()
        self.epochSamples += 1
        self.sampleQLock.release()
        
        return X, Y
        
      
#-------------------------------------------------------------------
# WorkerThread class fills queue with preprocessed training samples
#-------------------------------------------------------------------
class LoadingThread(threading.Thread):
    def __init__(self, threadID, list_IDs, imagePath, labelPath, normalization_args, augment, augmentation_params, numMaxTransforms, imageType, labelType, dim, crop_parameters, n_channels, n_classes, variableTypeX, variableTypeY, iQueue, sQueue, iQLock, sQLock, qSemaphore, stopFlag):
        threading.Thread.__init__(self)
            
        #ID
        self.threadID = str(threadID)
        
        #general settings
        self.list_IDs = list_IDs
        self.imagePath = imagePath
        self.labelPath = labelPath
        self.norm_args = normalization_args
        self.augment = augment
        self.numMaxTransforms = numMaxTransforms
        self.imageType = imageType
        self.labelType = labelType
        self.dim = dim
        self.crop_parameters = crop_parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.varTypeX = variableTypeX
        self.varTypeY = variableTypeY
        
        #queue and lock instances
        self.indexQueue = iQueue
        self.sampleQueue = sQueue
        self.indexQLock = iQLock
        self.sampleQLock = sQLock
        self.queueSemaphore = qSemaphore
        self.stopFlag = stopFlag
        
        #augmentor
        self.brainAugmentor = augmentor(**augmentation_params)
          
    
    def run(self):
        #print("Start Generator Thread " + self.threadID)
        
        #loop until stopflag is set
        while not self.stopFlag.is_set():
            #loop until no more samples to process
            while not self.indexQueue.empty():
            
                #get index to process (if empty in the meantime, release lock and return)
                try:
                    self.indexQLock.acquire()
                    ID = self.list_IDs[self.indexQueue.get()]
                    self.indexQLock.release()
                except queue.Empty:
                    self.indexQLock.release()
                    return
            
                #load and preprocess
                dataTuple = self.dataGeneration(ID)
            
                #wait for semaphore and put to queue
                self.queueSemaphore.acquire()
                self.sampleQLock.acquire()
                self.sampleQueue.put(dataTuple)
                self.sampleQLock.release()
                
        #print ("Exit Generator Thread " + self.threadID)
      
      
    def loadData(self, ID, folderPath, imgType):
        if '.nii.gz' in imgType or '.nii' in imgType:
            dataObj = self.load3DImagesNii(folderPath, ID, imgType, self.crop_parameters)[..., np.newaxis]
        elif imgType == '.dcm':
            dataObj = self.load3DImagesDcm(folderPath, ID, imgType)[..., np.newaxis]
        elif imgType == '.npy':
            pathToNpy = folderPath + '/' + ID + '.npy'
            dataObj = np.load(pathToNpy)[..., np.newaxis]
        else:
            raise Exception("Unknown image type")
        
        return dataObj
    
      
    def dataGeneration(self, ID):
        'Generates data containing one sample of batch'
        #Load data
        X_temp = self.loadData(ID, self.labelPath, self.labelType)
        Y_temp = self.loadData(ID, self.imagePath, self.imageType)

        #augmentation
        #----------------------------------------------------------------------
        if self.augment:
            X_temp, Y_temp = self.brainAugmentor.augmentXY(X_temp, Y_temp, self.numMaxTransforms)


        #preprocessing of X
        #----------------------------------------------------------------------
        # add noise
        if self.norm_args['addNoise']:
            Y_temp = addNoise(Y_temp, self.norm_args['normalization_threshold'], self.norm_args['meanNoiseDistribution'], self.norm_args['noiseMultiplicationFactor'])

        # simpleNormalization
        if self.norm_args['simpleNormalize']:
            Y_temp = simpleNormalization(Y_temp)
        
        # The intensity augmentation can only be used WITHOUt prior rescaling to [0,1] of [-1,1]!
        elif (self.norm_args['intensityNormalize'] == 1):
            Y_temp = intensityNormalization(Y_temp, augment=self.augment)
        
        # CTNormalization
        if self.norm_args['ctNormalize']:
            Y_temp = CTNormalization(Y_temp)

        # # gaussian filter
        # if self.norm_args['gaussian_filter']:
        #     Y_temp = gaussian_smoothing(Y_temp, self.norm_args['gaussian_filter_hsize'], self.norm_args['gaussian_filter_sigma'])
    
        #2D data: slice selection
        if len(self.dim) < 3:
            randint = np.random.randint(0, X_temp.shape[-2])
            X_temp = X_temp[...,randint,:]
            Y_temp = Y_temp[...,randint,:]

    


        #assign final data (add additional axis at index 0 for batch_dimension)
        #----------------------------------------------------------------------
        X = keras.utils.to_categorical(X_temp[..., 0], num_classes=self.n_classes, dtype=self.varTypeX)[np.newaxis, ...]
        Y = Y_temp[np.newaxis, ...]

        return X.astype(self.varTypeX), Y.astype(self.varTypeY)
    
    
    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_params):
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj)
        return image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]


    # load the DICOM files from a Folder, and return a 3 dim numpy array in LPS orientation
    def load3DImageDCM_LPS(self, pathToLoadDicomFolder, orientation):
    
        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        fileTuples = []

        # print('glob: {}'.format(pathToLoadDicomFolder))
        for fname in glob.glob(pathToLoadDicomFolder, recursive=False):
            fileTuples.append((fname,pydicom.dcmread(fname)))

        # sort the tuple (filename, dicom image) by image position
        if orientation == 1:
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[2])
            
        if orientation == 2:
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[0])

        if orientation == 3:
            fileTuples = sorted(fileTuples, key=lambda s: fileTuples[1].ImagePositionPatient[1])

        files = [x[1] for x in fileTuples]
        fileNames = [x[0] for x in fileTuples]

        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        # print(fileNames)

        img2d_shape = list(files[0].pixel_array.shape)

        if orientation == 0:
            L = len(files)
            P = img2d_shape[1] 
            S = img2d_shape[0] #in fact -
        if orientation == 1:
            L = img2d_shape[1]
            P = len(files)
            S = img2d_shape[0] #in fact -
            
        if orientation == 2:
            L = img2d_shape[1]
            P = img2d_shape[0]
            S = len(files)

        img_shape = [L,P,S]
        # print("image shape in LPS: {}".format(img_shape))

        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(files):
            img2d = s.pixel_array

            img_shape = [L,P,S]
            
            if orientation == 1:
                img3d[:, :, i] = img2d.T
            
            if orientation == 2:
                img2d = np.flip(img2d,axis=0)
                img3d[i, :, :] = img2d.T
        
            if orientation == 3:
                img2d = np.flip(img2d,axis=0)
                img3d[:, i, :] = img2d.T
                            
        return img3d


def get_orientation(dcm_path):
    """
    Function which returns the (encoded) orientation of a dicom volume.

    Input: Path to dicom volume

    Returns:  0 (if the images are sagittal slices), 1 (if the images are coronal slices), 2 (if the images are axial slices).
    """

    orientations = []

    for file in os.listdir(dcm_path):
        
        slc = dcm_path + '/' + file
        slc = pydicom.dcmread(slc)

        dicom_orientation = np.asarray(slc.ImageOrientationPatient).reshape((2,3))
        dicom_orientation_argmax = sorted(np.argmax(np.abs(dicom_orientation), axis=1))

        orientations.append(np.delete([0,1,2], sorted(dicom_orientation_argmax)))

    assert len(np.unique(orientations)) == 1, "ERROR: Not all slices have same orientation!"

    return np.unique(orientations)[0]


    def load3DImagesDcm(self, pathToDcmFolder, ID, crop_params):
        pathToDicomFolder = pathToDcmFolder + '/' + ID
        orientation = self.get_orientation(pathToDicomFolder)
        image_array = self.load3DImageDCM_LPS(pathToDicomFolder+'/*', orientation)
        return image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
    
