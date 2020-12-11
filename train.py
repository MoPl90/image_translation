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
        storage = mp['savmodpath'] + '/' + mp['model'].upper() + '/' + time.strftime("%Y%m%d_%H%M%S")  + '_'  + mp['comment']
    else:
        storage = out_folder
    
    model_path = storage + '/Models'
    prog_path = storage + '/res'
    script_path = storage + '/scripts'
    
    if not os.path.exists(storage):
        os.mkdir(storage)
        os.mkdir(model_path)
        os.mkdir(prog_path)
        os.mkdir(script_path)

    # Save the validation partition cases.
    f = open(storage + '/validation_partition.txt',"w")
    for p in partition['validation']:
        f.write(p + '\n')
    f.close()

    # Save all scripts
    working_dir = os.path.dirname(os.path.realpath("__file__"))
    copyfile(working_dir + '/' + config, script_path + '/'  + os.path.basename(config))

    for file in os.listdir(working_dir):
        try:
            if ".py" in file:
                copyfile(working_dir + '/' + file, script_path + '/' + file)
        except:
            pass

    return storage, prog_path, model_path


def collect_model_param(mp, model_params):

    if mp['2D'] == 0:
        params = {'image_shape': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start'], mp['channels']),
                'batchsize': mp['batchsize'],
                'batchsize_eval': mp['batchsize_eval'],
                'n_classes': mp['labels'],
                'lr': mp['lr'],
                'norm': mp['norm'],
                'dropoutrate': mp['dropoutrate'],
                'pool_size': mp['pool_size'],
                'out_dir': mp['savmodpath'] + '/' + mp['model'].upper() + '/' + time.strftime("%Y%m%d_%H%M%S")  + '_'  + mp['comment']}
    else:
        params = {'image_shape': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['channels']),
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

    if mp['2D'] == 0:
        params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']),
                'crop_parameters': [mp['x_start'], mp['x_end'], mp['y_start'], mp['y_end'], mp['z_start'], mp['z_end']],
                'n_classes': mp['labels'],
                'n_channels': mp['channels'],
                'batchsize': mp['batchsize'],
                'augment': mp['augment'],
                'n_classes': mp['labels']}
    else:
        params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start']),
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

    elif mp['model'].lower() == 'spade':
        model_param = collect_model_param(mp, spade_param)
        model = SPADE(**model_param)
        # partition = get_id_lists(mp['trnImgPath'], mp['validprop'], mp['shuffle'], gen_param['imgType'], mp['trnLabelPath'], gen_param['labelType'], gen_param['threshold'])
    

    #load pre-trained model
    if args.load > 0:
        model.load(args.load)
        print('Loaded model # %1d' % (args.load))


    #set up training an evaluation generators
    trainGenerator = DataGenerator(list_IDs=partition["train"], **gen_param_train, normalization_args=norm_param, augmentation_args=aug_param)
    testGenerator = DataGenerator(list_IDs=partition["validation"],  **gen_param_eval, normalization_args=norm_param, augmentation_args=aug_param)
    
    #Run training     
    while model.steps < mp['epochs'] * len(trainGenerator) * mp["batchsize"]:
        model.train_on_batch(trainGenerator, testGenerator)

    model.save(mp['epochs'])







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Load exiting model with given numerical identifier.", default=-1, type=int)
    parser.add_argument("-g", "--gpu", help="Select which gpu to use (0,1,2,3).", default='0', type=str)
    parser.add_argument("-c", "--config", help="Path to config file.", default='settings.cfg', type=str)
    parser.add_argument("-o", "--out", help="Output path for model etc..", default='', type=str)

    args = parser.parse_args()

    run_training(args)
