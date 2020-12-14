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
                imagePath, 
                labelPath, 
                normalization_args, 
                augment = 0, 
                augmentation_args = {}, 
                preload_data = False, 
                imageType = '.dcm', 
                labelType = '.dcm', 
                batchsize=32, 
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
        self.batch_size = batchsize
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.imagePath = imagePath
        self.labelPath = labelPath
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
            self.dataX = self.loadData(list_IDs, self.labelPath, self.labelType)
            self.dataY = self.loadData(list_IDs, self.imagePath, self.imageType)

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


    def loadData(self, IDs, folderPath, imgType):
        # X: we need to add a new axis for the channel dimension
        # Y: we need to transform to categorical
        #generate dict with sample-ID as key and dataobj as value
        dataDict = {}
        for i, ID in enumerate(IDs):
            if '.nii.gz' in imgType or '.nii' in imgType:
                dataObj = self.load3DImagesNii(folderPath, ID, imgType, self.crop_parameters)[..., np.newaxis]
            elif imgType == '.dcm':
                dataObj = self.load3DImagesDcm(folderPath, ID, imgType)[..., np.newaxis]
            elif imgType == '.npy':
                pathToNpy = folderPath + '/' + ID + '.npy'
                dataObj = np.load(pathToNpy)[..., np.newaxis]
            else:
                raise Exception("Unknown image type")
        
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
            self.dataX = self.loadData(list_IDs_temp, self.labelPath, self.labelType)
            self.dataY = self.loadData(list_IDs_temp, self.imagePath, self.imageType)

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

        return image_array[crop_parameters[0]: crop_parameters[1], crop_parameters[2]: crop_parameters[3], crop_parameters[4]: crop_parameters[5]]


    # load the DICOM files from a Folder, and return a 3 dim numpy array in LPS orientation
    def load3DImageDCM_LPS(self, pathToLoadDicomFolder, orientation):
    
        fileTuples = []

        for fname in glob.glob(pathToLoadDicomFolder, recursive=False):
            fileTuples.append((fname,pydicom.dcmread(fname)))

        # sort the tuple (filename, dicom image) by image position
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[orientation])

        files = [x[1] for x in fileTuples]
        fileNames = [x[0] for x in fileTuples]

        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        # print(fileNames)

        img2d_shape = list(files[0].pixel_array.shape)

        if orientation == 1:
            L = img2d_shape[1]
            P = img2d_shape[0]
            S = len(files)
        
        if orientation == 2:
            L = len(files)
            P = img2d_shape[1] 
            S = img2d_shape[0] #in fact -
            
        if orientation == 3:
            L = img2d_shape[1]
            P = len(files)
            S = img2d_shape[0] #in fact -

        img_shape = [L,P,S]
        # print("image shape in LPS: {}".format(img_shape))

        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(files):
            img2d = s.pixel_array

            img_shape = [L,P,S]
            
            if orientation == 0:
                img3d[i, :, :] = img2d.T
            
            if orientation == 1:
                img2d = np.flip(img2d,axis=0)
                img3d[:, i, :] = img2d.T
        
            if orientation == 2:
                img2d = np.flip(img2d,axis=0)
                img3d[:, :, i] = img2d.T
                            
        return img3d



    def load3DImagesDcm(self, pathToDcmFolder, ID, crop_params):
        pathToDicomFolder = pathToDcmFolder + '/' + ID
        orientation = get_Orientation(pathToDicomFolder)
        image_array = self.load3DImageDCM_LPS(pathToDicomFolder+'/*', orientation)
        return image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
    


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


def removeStrokeBelowThreshold(listOfCases, labelsPathFolder, image_type='.dcm', threshold=20, n_classes=1):

    valid = []
    for case in listOfCases:
        try:
            gts = nib.load(labelsPathFolder + '/' + case + image_type).get_fdata()
            if threshold >= 0 and np.sum(gts==n_classes-1) > threshold:
                valid.append(case)
            elif threshold < 0 and np.sum(gts == n_classes-1) < -threshold:
                valid.append(case)
        #If not gts file available -> normal database
        except FileNotFoundError:
            continue

    return valid

def get_id_lists(imagePathFolder, _validProportion, shuffle_train_val, image_type='.dcm', labelPathFolder=None, label_type='.nii.gz', threshold=0, n_classes=1):
    # generate List of train and valid IDs from the DicomFolderPaths randomly

    _listOfCases = listOfCasesInFolder(imagePathFolder, image_type)


    if labelPathFolder is not None:
        _listOfCases = removeStrokeBelowThreshold(_listOfCases, labelPathFolder, image_type=label_type, threshold=threshold, n_classes=n_classes)

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

    return np.unique(orientations)[0], np.asarray([np.sign(dicom_orientation[i][dicom_orientation_argmax[i]]) for i in range(2)], dtype=int)


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