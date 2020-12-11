from models import *
from data_generator import*
import argparse
import configparser
import json
import os
from shutil import copyfile
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np


def collect_parameters(cp, section):
    paramdict = {}
    
    for entry in cp[section]:
        if not any(i.isalpha() for i in cp[section][entry]):
            if '\n' in cp[section][entry]:
                paramdict.update({entry: [float(w) for w in cp[section][entry].split()]})
            elif '.' in cp[section][entry]:
                paramdict.update({entry: float(cp[section][entry])})
            else:
                paramdict.update({entry: int(cp[section][entry])})
        else:
            paramdict.update({entry: cp[section][entry]})
    
    return paramdict

def create_data_storage(mp, config, partition, out_folder):

    if out_folder=='':
        model_path = os.path.join(mp['savmodpath'] + '/' + mp['model'].upper() + '/' + time.strftime("%Y%m%d_%H%M%S")  + '_'  + mp['comment'] + '/model.h5')
    else:
        model_path = out_folder
    save_path   = os.path.dirname(model_path)
    log_path    = save_path + '/logs'
    script_path = save_path + '/scripts'


    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(log_path)
        os.mkdir(script_path)

    current_dir = os.path.dirname(os.path.realpath("__file__"))
    copyfile(current_dir + '/' + config, save_path + '/'  + config.split('/')[-1])

    # Save the validation partition cases.
    f = open(save_path+'/validation_partition.txt',"w")
    for p in partition['validation']:
        f.write(p + '\n')
    f.close()

    return save_path, log_path, model_path


def collect_model_param(mp, model_params):

    params = {'image_shape': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start'], mp['channels']),
              'batchsize': mp['batchsize'],
              'batchsize_eval': mp['batchsize_eval'],
              'n_classes': mp['labels'],
              'lr': mp['lr'],
              'norm': mp['norm'],
              'dropoutrate': mp['dropoutrate'],
              'pool_size': mp['pool_size'],
              'out_dir': mp['savmodpath'] + '/' + mp['model'].upper() + '/' + time.strftime("%Y%m%d_%H%M%S")  + '_'  + mp['comment']}


    params.update(model_params)
    return params

def assemble_generator_param(mp, gen_params, eval=False):

    params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']),
              'crop_parameters': [mp['x_start'], mp['x_end'], mp['y_start'], mp['y_end'], mp['z_start'], mp['z_end']],
              'n_classes': mp['labels'],
              'n_channels': mp['channels'],
              'batchsize': mp['batchsize'],
              'augment': mp['augment'],
              'n_classes': mp['labels']}

    if eval:
        params['batchsize'] = mp['batchsize_eval']
        params['augment'] = mp['augment_eval']

    params.update(gen_params)
    return params

def run_training(args):

    # Load config file.
    p = configparser.ConfigParser()
    p.optionxform = str
    p.read(args.config)
    
    # Set up graphics card settings.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    # Collect main, data generation, normalization, image, augmentation and callback settings.
    mp              = collect_parameters(p, 'MAIN')
    gen_param       = collect_parameters(p, 'GEN')
    norm_param      = collect_parameters(p, 'NORM')
    aug_param       = collect_parameters(p, 'AUG')
    gan_param       = collect_parameters(p, 'GAN')
    pix2pix_param   = collect_parameters(p, 'PIX2PIX')
    spade_param     = collect_parameters(p, 'SPADE')
    
    gen_param_train = assemble_generator_param(mp, gen_param, eval=False)
    gen_param_eval = assemble_generator_param(mp, gen_param, eval=True)

    # Generate data storage folders.    
    partition = get_id_lists(gen_param['imagePath'], 
                             mp['validprop'], 
                             mp['shuffle'], 
                             gen_param['imageType'], 
                             gen_param['labelPath'], 
                             gen_param['labelType'],
                             mp['lesion_threshold'])

    save_path, log_path, model_path = create_data_storage(mp, args.config, partition, args.out)

    #initialise the trainable model
    if mp['model'].lower() == 'gan':
        model_param = collect_model_param(mp, gan_param)
        model = GAN(**model_param)
       

    elif mp['model'].lower() == 'pix2pix':
        model_param = collect_model_param(mp, pix2pix_param)
        model = PIX2PIX(**model_param)
        # partition = get_id_lists(mp['trnImgPath'], mp['validprop'], mp['shuffle'], gen_param['imgType'], mp['trnLabelPath'], gen_param['labelType'], gen_param['threshold'])

    elif mp['model'].lower() == 'gan':
        model_param = collect_model_param(mp, spade_param)
        model = SPADE(**model_param)
        # partition = get_id_lists(mp['trnImgPath'], mp['validprop'], mp['shuffle'], gen_param['imgType'], mp['trnLabelPath'], gen_param['labelType'], gen_param['threshold'])
    

    #load pre-trained model
    if args.load > 0:
        model.load(args.load)
        print('Loaded model # %1d' % (args.load))


    trainGenerator = DataGenerator(list_IDs=partition["train"], **gen_param_train, normalization_args=norm_param, augmentation_args=aug_param)
                                    # imagePathFolder=args.image_path, 
                                    # labelPathFolder=args.seg_path, 
                                    # normalization_args=normalization_args, 
                                    # augment=1, 
                                    # augmentation_args=augmentation_args, 
                                    # preload_data = False, 
                                    # imageType = '.nii.gz', 
                                    # labelType = '_seg1.0.nii.gz', 
                                    # batch_size=kwargs["BATCH_SIZE"], 
                                    # dim=kwargs["image_shape"][:-1], 
                                    # crop_parameters=[0,128,0,128,0,96], 
                                    # n_channels=kwargs["image_shape"][-1], 
                                    # n_classes=kwargs["n_classes"],
                                    # shuffle=True)
                                    

    testGenerator = DataGenerator(list_IDs=partition["validation"],  **gen_param_eval, normalization_args=norm_param, augmentation_args=aug_param)
                                    # imagePathFolder=args.image_path, 
                                    # labelPathFolder=args.seg_path, 
                                    # normalization_args=normalization_args, 
                                    # augment=0, 
                                    # augmentation_args=augmentation_args, 
                                    # preload_data = False, 
                                    # imageType = '.nii.gz', 
                                    # labelType = '_seg1.0.nii.gz', 
                                    # batch_size=kwargs["BATCH_SIZE_EVAL"], 
                                    # dim=kwargs["image_shape"][:-1], 
                                    # crop_parameters=[0,128,0,128,0,96], 
                                    # n_channels=kwargs["image_shape"][-1], 
                                    # n_classes=kwargs["n_classes"],
                                    # shuffle=False)

    while model.steps < mp['epochs'] * len(trainGenerator) * mp["batchsize"]:
        model.train_on_batch(trainGenerator, testGenerator)

    model.save(args.epochs)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--seg_path", help="Path to segmentation data.", default="/scratch/mplatscher/data/DWI_data/training_data_nii_preprocessed/seg/", type=str)
    # parser.add_argument("-i", "--image_path", help="Path to image data.", default="/scratch/mplatscher/data/DWI_data/training_data_nii_preprocessed/ims/", type=str)
    # parser.add_argument("-e", "--epochs", help="Number of epochs to run.", default = 100, type=int)
    parser.add_argument("-l", "--load", help="Load exiting model with given numerical identifier.", default=-1, type=int)
    parser.add_argument("-g", "--gpu", help="Select which gpu to use (0,1,2,3).", default='0', type=str)
    # parser.add_argument("-t", "--toggle_mode", help="Toggle training mode; 0 = train on segmentations (default), 1 = train on images, 2 = train on differences", type=int, default=0)
    parser.add_argument("-c", "--config", help="Path to config file.", default='settings.cfg', type=str)
    parser.add_argument("-o", "--out", help="Output path for model etc..", default='', type=str)

    args = parser.parse_args()

    run_training(args)


    # # Set up graphics card settings.
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # #load the config file
    # with open('params.cfg', 'r') as file:
    #     kwargs = json.load(file)

    # #create output directory for the parameters in params.cfg
    # try:
    #     name = str(args.epochs) + 'epochs_'
    #     for k in kwargs.keys():
    #         name = name + k + str(kwargs[k]) + '_'
    #     name = name[:-1] #remove final underscore
    #     name = name.replace(", ", "_").replace("[", "").replace("]", "")
    #     os.mkdir(name)
    #     os.mkdir(name + '/res')
    #     os.mkdir(name + '/Models')
    #     os.popen('cp params.cfg ' + name + '/params.cfg')
    # except FileExistsError:
    #     print("Output directory already exists.")

    # if args.load > 0:
    #     model.load(args.load)
    #     print('Loaded model # %1d' % (args.load))
    # # #initialise the training and validation data generators for the two data domains
    # IDsList = get_id_lists(args.image_path, 0.15, 0, '.nii.gz', threshold=20)
    
    # normalization_args = {'simpleNormalize':0,
    #                       'addNoise':0,
    #                       'intensityNormalize':0,
    #                       'ctNormalize':1,
    #                       'gaussian_filter':0}

    # augmentation_args = {'maxNumberOfTransformation': 3,
    #                      'augment': True,
    #                      'flip': 1,
    #                      'rotationRangeXAxis': 5,
    #                      'rotationRangeYAxis': 5,
    #                      'rotationRangeZAxis': 5,
    #                      'zoomRange': 0.05,
    #                      'shiftXAxisRange': 0.02,
    #                      'shiftYAxisRange': 0.02,
    #                      'shiftZAxisRange': 0.02,
    #                      'stretchFactorXAxisRange': 0.05,
    #                      'stretchFactorYAxisRange': 0.05,
    #                      'stretchFactorZAxisRange': 0.05,
    #                      'shear_NormalXAxisRange': 0.03,
    #                      'shear_NormalYAxisRange': 0.03,
    #                      'shear_NormalZAxisRange': 0.03}


    # trainGenerator = DataGenerator(list_IDs=IDsList["train"], 
    #                                 imagePathFolder=args.image_path, 
    #                                 labelPathFolder=args.seg_path, 
    #                                 normalization_args=normalization_args, 
    #                                 augment=1, 
    #                                 augmentation_args=augmentation_args, 
    #                                 preload_data = False, 
    #                                 imageType = '.nii.gz', 
    #                                 labelType = '_seg1.0.nii.gz', 
    #                                 batch_size=kwargs["BATCH_SIZE"], 
    #                                 dim=kwargs["image_shape"][:-1], 
    #                                 crop_parameters=[0,128,0,128,0,96], 
    #                                 n_channels=kwargs["image_shape"][-1], 
    #                                 n_classes=kwargs["n_classes"],
    #                                 shuffle=True)

    # testGenerator = DataGenerator(list_IDs=IDsList["validation"], 
    #                                 imagePathFolder=args.image_path, 
    #                                 labelPathFolder=args.seg_path, 
    #                                 normalization_args=normalization_args, 
    #                                 augment=0, 
    #                                 augmentation_args=augmentation_args, 
    #                                 preload_data = False, 
    #                                 imageType = '.nii.gz', 
    #                                 labelType = '_seg1.0.nii.gz', 
    #                                 batch_size=kwargs["BATCH_SIZE_EVAL"], 
    #                                 dim=kwargs["image_shape"][:-1], 
    #                                 crop_parameters=[0,128,0,128,0,96], 
    #                                 n_channels=kwargs["image_shape"][-1], 
    #                                 n_classes=kwargs["n_classes"],
    #                                 shuffle=False)

    # while model.steps < args.epochs * len(trainGenerator) * kwargs["BATCH_SIZE"]:
    #     model.train_on_batch(trainGenerator, testGenerator)

    # model.save(args.epochs)
