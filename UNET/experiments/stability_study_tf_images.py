# Model training per stability scheme and evaluation on example images

from sklearn.model_selection import KFold

import os
import os.path as op
import time
import glob
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import atexit
import pandas as pd

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics, ssim, sssim, psnr
from learning_wavelets.models.unet import unet

from tf_fits.image import image_decode_fits
from astropy.io import fits

CUDA_VISIBLE_DEVICES="-1"

dataset_lr = []
dataset_hr = []


for filename in glob.glob('/home/julia/understanding-unets-master/hr_images/*.fits'):
    fits_file = filename
    header = 0
    img = tf.io.read_file(fits_file)
    img = image_decode_fits(img, header)
    dataset_hr.append(img)
    
for filename in glob.glob('/home/julia/understanding-unets-master/rl_images/rl_images_10_0015/*.fits'):
    fits_file = filename
    header = 0
    img = tf.io.read_file(fits_file)
    img = image_decode_fits(img, header)
    dataset_lr.append(img)

# define images
img_20042790 = np.array(dataset_lr[919])
img_20042790 = img_20042790.reshape(1,256,256)
    
img_20043513 = np.array(dataset_lr[638])
img_20043513 = img_20043513.reshape(1,256,256)

img_20160226 = np.array(dataset_lr[1210])
img_20160226 = img_20160226.reshape(1,256,256)

img_20042790_hr = np.array(dataset_hr[919])
img_20042790_hr = img_20042790_hr.reshape(256,256,1)
    
img_20043513_hr = np.array(dataset_hr[638])
img_20043513_hr = img_20043513_hr.reshape(256,256,1)

img_20160226_hr = np.array(dataset_hr[1210])
img_20160226_hr = img_20160226_hr.reshape(256,256,1)


def train_unet(tf_dataset, im_ds_val, base_n_filters=64):

    # model definition
    run_params = {
        'n_layers': 2,#1,#5,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(2)],#[base_n_filters * 2**i for i in range(5)],
        'layers_n_non_lins': 0,#2,#2, 
        'non_relu_contract': False,
        'bn': True,
    }
    n_epochs = 50
    run_id = f'unet_{base_n_filters}_dynamic_st_{int(time.time())}'
    #chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-30.hdf5'

    print(chkpt_path)

    # callbacks preparation
    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    n_channels = 1
    # run distributed
    mirrored_strategy = tf.distribute.MirroredStrategy()
    atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore

    with mirrored_strategy.scope():
        model = unet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    print(model.summary(line_length=114))

    # actual training
    model.fit(
        tf_dataset,
        #steps_per_epoch=2,#10,
        epochs=n_epochs,
        validation_data=im_ds_val,
        #validation_steps=1,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback, lrate_cback],
        shuffle=False,
    )

    return '/home/julia/understanding-unets-master/experiments' + chkpt_path[1:]

n_folds = 5
k_folds = 10

nfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print('**************************')

n_fold = 0

for n_train_index, n_test_index in nfold.split(dataset_hr):
    X_train = [dataset_hr[i][:][:] for i in list(n_train_index)]
    X_test = [dataset_hr[i][:][:] for i in list(n_test_index)]
    y_train = [dataset_lr[i][:][:] for i in list(n_train_index)]
    y_test = [dataset_lr[i][:][:] for i in list(n_test_index)]
    
    #X = tf.data.Dataset.from_tensors((X_train, y_train))
    X_test = tf.data.Dataset.from_tensors((X_test, y_test))

    # define folds for next split
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    k_fold = 0
    print("CV FOLDS: ", n_fold, k_fold)
    for k_train_index, k_test_index in kfold.split(X_train):

        # define array for results
        results_psnr = np.zeros((3,1))
        results_ssim = np.zeros((3,1))
        results_sssim = np.zeros((3,1))
        
        print('---------------')

        X_train_k = [X_train[i][:][:] for i in list(k_train_index)]
        X_test_k = [X_train[i][:][:] for i in list(k_test_index)]
        y_train_k = [y_train[i][:][:] for i in list(k_train_index)]
        y_test_k = [y_train[i][:][:] for i in list(k_test_index)]
    
        X_train_k = tf.data.Dataset.from_tensors((X_train_k, y_train_k))
        X_val_k = tf.data.Dataset.from_tensors((X_test_k, y_test_k))

        base_n_filters = 64

        run_params = {
            'n_layers': 2,#1,#5,
            'pool': 'max',
            "layers_n_channels": [base_n_filters * 2**i for i in range(2)],#[base_n_filters * 2**i for i in range(5)],
            'layers_n_non_lins': 0,#2,#2, 
            'non_relu_contract': False,
            'bn': True,
        }

        k_fold_id = train_unet(X_train_k, X_val_k)
        n_channels = 1
        model = unet(input_size=(None, None, n_channels), **run_params)
        #model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
        model.load_weights(k_fold_id)
        
        # img 20042790
        y_pred_20042790 = model.predict(img_20042790)
        y_pred_20042790 = y_pred_20042790.reshape(1,256,256)

        # save predicted image
        img_name = "img_results_10_0015/20042790_" + str(n_fold) +  "_" + str(k_fold) + ".fits"
        hdu = fits.PrimaryHDU(y_pred_20042790)#.detach().cpu().numpy())
        hdu.writeto(img_name, overwrite=True)

        y_pred_20042790 = y_pred_20042790.reshape(256,256,1)
        results_psnr[0, 0] = psnr(y_pred_20042790, img_20042790_hr)
        results_ssim[0, 0] = ssim(y_pred_20042790, img_20042790_hr)
        results_sssim[0, 0] = sssim(y_pred_20042790, img_20042790_hr)

        # img 20043513
        y_pred_20043513 = model.predict(img_20043513)
        y_pred_20043513 = y_pred_20043513.reshape(1,256,256)

        # save predicted image
        img_name = "img_results_10_0015/20043513_" + str(n_fold) +  "_" + str(k_fold) + ".fits"
        hdu = fits.PrimaryHDU(y_pred_20043513)#.detach().cpu().numpy())
        hdu.writeto(img_name, overwrite=True)

        results_psnr[1, 0] = psnr(y_pred_20043513.reshape(256,256,1), img_20043513_hr)
        results_ssim[1, 0] = ssim(y_pred_20043513.reshape(256,256,1), img_20043513_hr)
        results_sssim[1, 0] = sssim(y_pred_20043513.reshape(256,256,1), img_20043513_hr)

        # img 20160226
        y_pred_20160226 = model.predict(img_20160226)
        y_pred_20160226 = y_pred_20160226.reshape(1,256,256)

        # save predicted image
        img_name = "img_results_10_0015/20160226_" + str(n_fold) +  "_" + str(k_fold) + ".fits"
        hdu = fits.PrimaryHDU(y_pred_20160226)#.detach().cpu().numpy())
        hdu.writeto(img_name, overwrite=True)

        results_psnr[2, 0] = psnr(y_pred_20160226.reshape(256,256,1), img_20160226_hr)
        results_ssim[2, 0] = ssim(y_pred_20160226.reshape(256,256,1), img_20160226_hr)
        results_sssim[2, 0] = sssim(y_pred_20160226.reshape(256,256,1), img_20160226_hr)

        results_psnr_df = pd.DataFrame(results_psnr)
        results_ssim_df = pd.DataFrame(results_ssim)
        results_sssim_df = pd.DataFrame(results_sssim)
     
        psnr_name = "img_results_10_0015/results_psnr_" + str(n_fold) + "_" + str(k_fold) + ".csv"
        ssim_name = "img_results_10_0015/results_ssim_" + str(n_fold) + "_" + str(k_fold) + ".csv"
        sssim_name = "img_results_10_0015/results_sssim_" + str(n_fold) + "_" + str(k_fold) + ".csv"

        # saving the dataframe
        results_psnr_df.to_csv(psnr_name)
        results_ssim_df.to_csv(ssim_name)
        results_sssim_df.to_csv(sssim_name)

        k_fold += 1
    n_fold += 1
