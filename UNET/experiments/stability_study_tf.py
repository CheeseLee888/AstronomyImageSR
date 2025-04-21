# Model training and evaluation per stability scheme

from sklearn.model_selection import KFold

import os
import os.path as op
import time
import glob
import sys
import pandas as pd
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import atexit

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics, ssim, sssim, psnr
from learning_wavelets.models.unet import unet

from tf_fits.image import image_decode_fits

CUDA_VISIBLE_DEVICES="0,1,2"

dataset_lr = []
dataset_hr = []


for filename in glob.glob('/home/julia/understanding-unets-master/hr_images/*.fits'):
    fits_file = filename
    header = 0
    img = tf.io.read_file(fits_file)
    img = image_decode_fits(img, header)
    img = tf.reshape(img, [256, 256, 1])
    dataset_hr.append(img)
    
for filename in glob.glob('/home/julia/understanding-unets-master/rl_images/rl_images_10/*.fits'):
    fits_file = filename
    header = 0
    img = tf.io.read_file(fits_file)
    img = image_decode_fits(img, header)
    img = tf.reshape(img, [256, 256, 1])
    dataset_lr.append(img)

def train_unet(tf_dataset, im_ds_val, base_n_filters=64):

    # model definition
    run_params = {
        'n_layers': 2,#1,#2,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(2)],#[base_n_filters * 2**i for i in range(2)],
        'layers_n_non_lins': 0,#2,#0, 
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

def evaluate_old_unet(dataset,
        run_id,
        n_epochs=50,
        base_n_filters=64, 
        n_layers=2,#2,
        layers_n_non_lins=0,#0,
        n_samples=None,
    ):
        
    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': layers_n_non_lins,
        'non_relu_contract': False,
        'bn': True,
    }
    
    n_channels = 1
    model = unet(input_size=(None, None, n_channels), **run_params)
        
    #model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    model.load_weights(run_id)

    val_set = dataset

    for x, y_true in tqdm(val_set.as_numpy_iterator()):
        y_pred = model.predict(x)
        #y_true = y_true.reshape(y_true.shape[0],y_true.shape[1], y_true.shape[2], 1)
        #y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[1], y_pred.shape[2], 1)
        sssim_vals= sssim(y_true, y_pred)
        ssim_vals= ssim(y_true, y_pred)
        psnr_vals= psnr(y_true, y_pred)

    return sssim_vals, ssim_vals, psnr_vals

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

    # save fold results
    val_results_psnr = np.zeros((1262,50))
    val_results_ssim = np.zeros((1262,50))
    val_results_sssim = np.zeros((1262,50))

    train_results_psnr = np.zeros((1262,50))
    train_results_ssim = np.zeros((1262,50))
    train_results_sssim = np.zeros((1262,50))

    test_results_psnr = np.zeros((1262,50))
    test_results_ssim = np.zeros((1262,50))
    test_results_sssim = np.zeros((1262,50))

    # define folds for next split
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    k_fold = 0
    print("CV FOLDS: ", n_fold, k_fold)
    for k_train_index, k_test_index in kfold.split(X_train):
        
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

        # Evaluation on validation set
        eval_sssim, eval_ssim, eval_psnr = evaluate_old_unet(dataset=X_val_k, run_id=k_fold_id)

        val_results_psnr[0:len(eval_psnr),k_fold] = eval_psnr
        val_results_ssim[0:len(eval_ssim),k_fold] = eval_ssim
        val_results_sssim[0:len(eval_sssim),k_fold] = eval_sssim

        # Evaluation on training set
        train_sssim, train_ssim, train_psnr = evaluate_old_unet(dataset=X_train_k, run_id=k_fold_id)

        train_results_psnr[0:len(train_psnr),k_fold] = train_psnr
        train_results_ssim[0:len(train_ssim),k_fold] = train_ssim
        train_results_sssim[0:len(train_sssim),k_fold] = train_sssim

    
        # Evaluation on test set
        test_sssim, test_ssim, test_psnr = evaluate_old_unet(dataset=X_test, run_id=k_fold_id)
        #print(test_sssim)

        test_results_psnr[0:len(test_psnr),k_fold] = test_psnr
        test_results_ssim[0:len(test_ssim),k_fold] = test_ssim
        test_results_sssim[0:len(test_sssim),k_fold] = test_sssim

        k_fold += 1

    val_results_psnr_df = pd.DataFrame(val_results_psnr)
    val_results_ssim_df = pd.DataFrame(val_results_ssim)
    val_results_sssim_df = pd.DataFrame(val_results_sssim)
     
    val_psnr_name = "results_10/val_results_psnr_" + str(n_fold) + ".csv"
    val_ssim_name = "results_10/val_results_ssim_" + str(n_fold) + ".csv"
    val_sssim_name = "results_10/val_results_sssim_" + str(n_fold) + ".csv"

    # saving the dataframe
    val_results_psnr_df.to_csv(val_psnr_name)
    val_results_ssim_df.to_csv(val_ssim_name)
    val_results_sssim_df.to_csv(val_sssim_name)

    train_results_psnr_df = pd.DataFrame(train_results_psnr)
    train_results_ssim_df = pd.DataFrame(train_results_ssim)
    train_results_sssim_df = pd.DataFrame(train_results_sssim)
     
    train_psnr_name = "results_10/train_results_psnr_" + str(n_fold) + ".csv"
    train_ssim_name = "results_10/train_results_ssim_" + str(n_fold) + ".csv"
    train_sssim_name = "results_10/train_results_sssim_" + str(n_fold) + ".csv"

    # saving the dataframe
    train_results_psnr_df.to_csv(train_psnr_name)
    train_results_ssim_df.to_csv(train_ssim_name)
    train_results_sssim_df.to_csv(train_sssim_name)

    test_results_psnr_df = pd.DataFrame(test_results_psnr)
    test_results_ssim_df = pd.DataFrame(test_results_ssim)
    test_results_sssim_df = pd.DataFrame(test_results_sssim)
     
    test_psnr_name = "results_10/test_results_psnr_" + str(n_fold) + ".csv"
    test_ssim_name = "results_10/test_results_ssim_" + str(n_fold) + ".csv"
    test_sssim_name = "results_10/test_results_sssim_" + str(n_fold) + ".csv"

    test_results_psnr_df.to_csv(test_psnr_name)
    test_results_ssim_df.to_csv(test_ssim_name)
    test_results_sssim_df.to_csv(test_sssim_name)


    n_fold += 1


