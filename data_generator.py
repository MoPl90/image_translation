from tensorflow.keras import backend as K

import numpy as np
import tensorflow.keras as keras
import pydicom
import glob
from PIL import Image
from augmentation import augmentor
import random
import datetime
import os
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.utils import to_categorical, Sequence
from skimage.transform import resize

class DataGenerator(Sequence):

    def __init__(self, 
                list_IDs, 
                imagePathFolder, 
                labelPathFolder, 
                normalization_args, 
                augment, 
                augmentation_args, 
                preload_data = False, 
                imageType = '.dcm', 
                labelType = '.dcm', 
                batch_size=32, 
                dim=(32, 32, 32), 
                crop_parameters=[0,10,0,10,0,10], 
                resize_parameters=None,
                n_channels=1, 
                n_classes=10, 
                shuffle=True, 
                variableTypeX = 'float32', 
                variableTypeY = 'float32', 
                savePath = None, 
                preload=False):

        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.imagePathFolder = imagePathFolder
        self.labelPathFolder = labelPathFolder
        self.imageType = imageType
        self.labelType = labelType
        self.variableTypeX = variableTypeX
        self.variableTypeY = variableTypeY
        self.crop_parameters = crop_parameters
        self.resize_parameters = resize_parameters

        self.augment = augment
        self.augmentation_args = augmentation_args
        self.normalization_args = normalization_args
        self.save_path = savePath
        self.numberOfAufmentationImageSaved = 0

        #preload data into RAM if stated
        self.preload = preload_data
        if self.preload:
            self.dataX = self._preloadData(list_IDs, self.labelPathFolder, self.labelType)
            self.dataY = self._preloadData(list_IDs, self.imagePathFolder, self.imageType)

        #create augmentor
        augmentation_parameters = {
                    'rotationAngle_Xaxis': self.augmentation_args['rotationRangeXAxis'],
                    'rotationAngle_Yaxis': self.augmentation_args['rotationRangeYAxis'],
                    'rotationAngle_Zaxis': self.augmentation_args['rotationRangeZAxis'],
                    'shift_Xaxis': self.augmentation_args['shiftXAxisRange'],
                    'shift_Yaxis': self.augmentation_args['shiftYAxisRange'],
                    'shift_Zaxis': self.augmentation_args['shiftZAxisRange'],
                    'stretch_Xaxis': self.augmentation_args['stretchFactorXAxisRange'],
                    'stretch_Yaxis': self.augmentation_args['stretchFactorYAxisRange'],
                    'stretch_Zaxis': self.augmentation_args['stretchFactorZAxisRange'],
                    'shear_NormalXAxis': self.augmentation_args['shear_NormalXAxisRange'],
                    'shear_NormalYAxis': self.augmentation_args['shear_NormalYAxisRange'],
                    'shear_NormalZAxis': self.augmentation_args['shear_NormalZAxisRange'],
                    'zoom': self.augmentation_args['zoomRange'],
                    'flip': self.augmentation_args['flip'],
                    }

        self.brainAugmentor = augmentor(**augmentation_parameters)

        #set index order for first epoch
        self.on_epoch_end()


    def _preloadData(self, IDs, folderPath, imgType):
        # X: we need to add a new axis for the channel dimension
        # Y: we need to transform to categorical
        #generate dict with sample-ID as key and dataobj as value
        dataDict = {}
        for i, ID in enumerate(IDs):
            #load data
            dataObj = self.load3DImagesNii(folderPath, ID, imgType, self.crop_parameters, self.resize_parameters)[..., np.newaxis]

            #add new key-value pair
            dataDict[ID] = dataObj

        return dataDict


    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
        self.n = 0

        #Print mean number of augmentations per input sample
        print('; {:.1f} augmentations performed on average'.format(self.brainAugmentor.meanNumberTransforms()))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        #Load data at this point if not preloaded already
        if not self.preload:
            self.dataX = self._preloadData(list_IDs_temp, self.labelPathFolder, self.labelType)
            self.dataY = self._preloadData(list_IDs_temp, self.imagePathFolder, self.imageType)

        #Generate data
        X = np.empty((self.batch_size, *self.dim, self.n_classes))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, ID in enumerate(list_IDs_temp):
            #get according data
            X_temp = self.dataX[ID]
            Y_temp = self.dataY[ID]

            #augmentation
            #-------------------
            if (self.augment == 1):
                X_temp, Y_temp = self.brainAugmentor.augmentXY(X_temp, Y_temp, self.augmentation_args['maxNumberOfTransformation'])

            #preprocessing of Y
            #-------------------
            # add noise
            if (self.normalization_args['addNoise'] == 1):
                Y_temp = addNoise(Y_temp, self.normalization_args['normalization_threshold'], self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])

            # simpleNormalization
            if (self.normalization_args['simpleNormalize'] == 1):
                Y_temp = simpleNormalization(Y_temp)

            # The intensity augmentation can only be used WITHOUt prior rescaling to [0,1] of [-1,1]!
            elif (self.normalization_args['intensityNormalize'] == 1):
                Y_temp = intensityNormalization(Y_temp, augment=self.augment)

            # CTNormalization
            if (self.normalization_args['ctNormalize'] == 1):
                Y_temp = ctNormalization(Y_temp)

            # gaussian filter
            if (self.normalization_args['gaussian_filter'] == 1):
                Y_temp = gaussian_smoothing(Y_temp, self.normalization_args['gaussian_filter_hsize'], self.normalization_args['gaussian_filter_sigma'])

            # #Rearrage arrays in dicom ordering, with slice dim first
            # X_temp = np.transpose(X_temp,(2,1,0,3))[:, ::-1,...]
            # Y_temp = np.transpose(Y_temp,(2,1,0,3))[:, ::-1,...]

            #2D data: slice selection
            if len(self.dim) < 3:
                randint = np.random.randint(0, X_temp.shape[-2], self.batch_size)
                X_temp = X_temp[...,randint[i],:]
                Y_temp = Y_temp[...,randint[i],:]

            #assign final data
            X[i,] = to_categorical(X_temp[...,0], num_classes=self.n_classes, dtype=self.variableTypeY)
            Y[i,] = Y_temp

        return X.astype(self.variableTypeX), Y.astype(self.variableTypeY)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        # Find list of IDs
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

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
        
    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_parameters, resize_parameters=None):
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj)

        if resize_parameters is not None:
            image_array = resize(image_array, resize_parameters)

        return image_array[crop_parameters[0]: crop_parameters[1], crop_parameters[2]: crop_parameters[3], crop_parameters[4]: crop_parameters[5]]



#####################################
# Helper functions
#####################################

def listOfCasesInFolder(pathToFolder, image_type='.dcm'):
    listOfImages = []
    listOfFiles = os.listdir(pathToFolder)
    for f in listOfFiles:
        if f.endswith(image_type) and f[0] != '.':
            listOfImages.append(f.split('.')[0])
    
    return list(listOfImages)


def removeStrokeBelowThreshold(listOfCases, labelsPathFolder, image_type='.dcm', threshold=20):

    valid = []
    for case in listOfCases:
        gts = nib.load(labelsPathFolder + '/' + case + image_type).get_fdata()

        if threshold >= 0 and np.sum(gts) >= threshold:# or np.sum(gts) < -1. * threshold:
            valid.append(case)
        elif threshold < 0 and np.sum(gts) < -threshold:
            valid.append(case)

    return valid

def get_id_lists(imagePathFolder, _validProportion, shuffle_train_val, image_type='.dcm', labelPathFolder=None, label_type='.nii.gz', threshold=0):
    # generate List of train and valid IDs from the DicomFolderPaths randomly

    _listOfCases = listOfCasesInFolder(imagePathFolder, image_type)


    if labelPathFolder is not None:
        _listOfCases = removeStrokeBelowThreshold(_listOfCases, labelPathFolder, image_type=label_type, threshold=threshold)

    index = np.arange(len(_listOfCases))

    if shuffle_train_val:
        np.random.shuffle(index)

    validNumbers = int(np.floor(len(_listOfCases) * _validProportion))

    indexValid = random.sample(range(0, len(_listOfCases)), validNumbers)
    indexTrain = []
    for k in index:
        if k not in indexValid:
            indexTrain.append(k)

    _listOfTrainCasesID = [_listOfCases[k] for k in indexTrain]
    _listOfValidCasesID = [_listOfCases[k] for k in indexValid]

    partition = {'train': _listOfTrainCasesID, 'validation': _listOfValidCasesID}

    return partition

    
def simpleNormalization(array):
    vol = array

    if np.amax(vol) > 255:
        vol[vol < 30] = 0
    else:
        vol -= np.max(vol[0:20,:,-1])

    vol = np.clip(vol,0,np.percentile(vol[35:100, 35:100, 6:32],99.5))
    # vol /= np.max(vol)

    vol -= np.mean(vol)
    vol /= np.std(vol)
    vol = -1 + 2 * (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

    return vol

def ctNormalization(array):
    
    vol = np.clip(array, 0.001, 100)

    vol -= np.mean(vol)
    vol /= np.std(vol)
    vol = -1 + 2 * (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

    return vol

#####################################
# Testing routine
#####################################

if __name__ == "__main__":
    import matplotlib.pylab as plt
    import time
    import seaborn as sns

    IDsList = get_id_lists("/scratch/mplatscher/data/DWI_data/nii_data_skull_stripped/", 0.15, 0, '.nii.gz')
    
    augmentation_args = {'maxNumberOfTransformation':3,
                         'flip': 1,
                         'rotationRangeXAxis': 5,
                         'rotationRangeYAxis': 5,
                         'rotationRangeZAxis': 5,
                         'zoomRange': 0.05,
                         'shiftXAxisRange': 0.02,
                         'shiftYAxisRange': 0.02,
                         'shiftZAxisRange': 0.02,
                         'stretchFactorXAxisRange': 0.05,
                         'stretchFactorYAxisRange': 0.05,
                         'stretchFactorZAxisRange': 0.05,
                         'shear_NormalXAxisRange': 0.03,
                         'shear_NormalYAxisRange': 0.03,
                         'shear_NormalZAxisRange': 0.03}

    normalization_args = {'simpleNormalize':1,
                          'addNoise':0,
                          'intensityNormalize':0,
                          'ctNormalize':0,
                          'gaussian_filter':0}

    trainGenerator = DataGenerator(list_IDs=IDsList["train"], 
                                  imagePathFolder="/scratch/mplatscher/data/DWI_data/nii_data_skull_stripped/", 
                                  labelPathFolder="/scratch/mplatscher/data/DWI_data/training_data_processed_segmentation_28classes/", 
                                  normalization_args=normalization_args, 
                                  augment=1, 
                                  augmentation_args=augmentation_args, 
                                  preload_data = False, 
                                  imageType = '.nii.gz', 
                                  labelType = '_aseg.nii.gz', 
                                  batch_size=16, 
                                  dim=(128,128), 
                                  crop_parameters=[0,128,0,128,4,36], 
                                  n_channels=1, 
                                  n_classes=28,
                                  shuffle=True)

    testGenerator = DataGenerator(list_IDs=IDsList["validation"], 
                                  imagePathFolder="/scratch/mplatscher/data/DWI_data/nii_data_skull_stripped/", 
                                  labelPathFolder="/scratch/mplatscher/data/DWI_data/training_data_processed_segmentation_28classes/", 
                                  normalization_args=normalization_args, 
                                  augment=1, 
                                  augmentation_args=augmentation_args, 
                                  preload_data = False, 
                                  imageType = '.nii.gz', 
                                  labelType = '_aseg.nii.gz', 
                                  batch_size=1, 
                                  dim=(128,128), 
                                  crop_parameters=[0,128,0,128,4,36], 
                                  n_channels=1, 
                                  n_classes=28,
                                  shuffle=True)


    n_classes = 28
    cols = sns.color_palette("pastel", n_classes - 1)
    # background is transparent
    cols.insert(0, 'none')
    # Add stroke class
    cols.append('r')


    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    ax = ax.flatten()

    tic = time.perf_counter()
    imA, imB = next(testGenerator)
    toc = time.perf_counter()
    print(f"Data generation took {toc - tic:0.4f} seconds")

    if len(testGenerator.dim) > 2:
        imA = imA[0]
        imB = imB[0]
        init = 3
    else:
        init = 0
        
    i = init
    for a in ax:
        a.imshow(imB[i,:,:,0], cmap='bone', vmin=-1, vmax=1)
        a.contourf(np.argmax(imA[i,:,:], axis=-1), levels = np.arange(testGenerator.n_classes+1)-0.5, colors=cols, alpha=0.3)
        a.set_xticks([],[])
        a.set_yticks([],[])
        i+=1
    plt.savefig('testA.png')

    fig, ax = plt.subplots(4, 4, figsize=(16, 16))

    ax = ax.flatten()
    i = init
    for a in ax:
        a.imshow(imB[i,:,:,0], cmap='bone', vmin=-1, vmax=1)

        a.contour(np.argmax(imA[i,:,:,:], axis=-1), levels = np.arange(testGenerator.n_classes+1)-0.5, colors=cols, linewidths=0.5, alpha=0.75)
        a.set_xticks([],[])
        a.set_yticks([],[])
        i+=1
    plt.savefig('testB.png')