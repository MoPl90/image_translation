from models import *
from data_generator import*
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seg_path", help="Path to segmentation data.", default="/scratch/mplatscher/data/DWI_data/training_data_nii_preprocessed/seg/", type=str)
    parser.add_argument("-i", "--image_path", help="Path to image data.", default="/scratch/mplatscher/data/DWI_data/training_data_nii_preprocessed/ims/", type=str)
    parser.add_argument("-e", "--epochs", help="Number of epochs to run.", default = 100, type=int)
    parser.add_argument("-l", "--load", help="Load exiting model with given numerical identifier.", default=-1, type=int)
    parser.add_argument("-g", "--gpu", help="Select which gpu to use (0,1,2,3).", default='0', type=str)
    parser.add_argument("-t", "--toggle_mode", help="Toggle training mode; 0 = train on segmentations (default), 1 = train on images, 2 = train on differences", type=int, default=0)
    args = parser.parse_args()


    # Set up graphics card settings.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #load the config file
    with open('params.cfg', 'r') as file:
        kwargs = json.load(file)

    #create output directory for the parameters in params.cfg
    try:
        name = str(args.epochs) + 'epochs_'
        for k in kwargs.keys():
            name = name + k + str(kwargs[k]) + '_'
        name = name[:-1] #remove final underscore
        name = name.replace(", ", "_").replace("[", "").replace("]", "")
        os.mkdir(name)
        os.mkdir(name + '/res')
        os.mkdir(name + '/Models')
        os.popen('cp params.cfg ' + name + '/params.cfg')
    except FileExistsError:
        print("Output directory already exists.")

    #initialise the trainable model
    kwargs['out_dir'] = name
    model = trainableModel(**kwargs, mode=args.toggle_mode)

    if args.load > 0:
        model.load(args.load)
        print('Loaded model # %1d' % (args.load))
    # #initialise the training and validation data generators for the two data domains
    IDsList = get_id_lists(args.image_path, 0.15, 0, '.nii.gz', threshold=20)
    
    normalization_args = {'simpleNormalize':0,
                          'addNoise':0,
                          'intensityNormalize':0,
                          'ctNormalize':1,
                          'gaussian_filter':0}

    augmentation_args = {'maxNumberOfTransformation': 3,
                         'augment': True,
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


    trainGenerator = DataGenerator(list_IDs=IDsList["train"], 
                                    imagePathFolder=args.image_path, 
                                    labelPathFolder=args.seg_path, 
                                    normalization_args=normalization_args, 
                                    augment=1, 
                                    augmentation_args=augmentation_args, 
                                    preload_data = False, 
                                    imageType = '.nii.gz', 
                                    labelType = '_seg1.0.nii.gz', 
                                    batch_size=kwargs["BATCH_SIZE"], 
                                    dim=kwargs["image_shape"][:-1], 
                                    crop_parameters=[0,128,0,128,0,96], 
                                    n_channels=kwargs["image_shape"][-1], 
                                    n_classes=kwargs["n_classes"],
                                    shuffle=True)

    testGenerator = DataGenerator(list_IDs=IDsList["validation"], 
                                    imagePathFolder=args.image_path, 
                                    labelPathFolder=args.seg_path, 
                                    normalization_args=normalization_args, 
                                    augment=0, 
                                    augmentation_args=augmentation_args, 
                                    preload_data = False, 
                                    imageType = '.nii.gz', 
                                    labelType = '_seg1.0.nii.gz', 
                                    batch_size=kwargs["BATCH_SIZE_EVAL"], 
                                    dim=kwargs["image_shape"][:-1], 
                                    crop_parameters=[0,128,0,128,0,96], 
                                    n_channels=kwargs["image_shape"][-1], 
                                    n_classes=kwargs["n_classes"],
                                    shuffle=False)

    while model.steps < args.epochs * len(trainGenerator) * kwargs["BATCH_SIZE"]:
        model.train_on_batch(trainGenerator, testGenerator)

    model.save(args.epochs)
