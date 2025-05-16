# file to download image results from the trained models for 3 example images from hold-out test set
# to perform local image stability study

# not customed yet

import os
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from tqdm import tqdm
from models import SRCNN
# from utils import calc_psnr, calc_ssim, calc_sssim
import numpy as np

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename 

import sys
sys.path.append("/Users/peterli/Desktop/BS_thesis/code_main/Data Preparation")  # locally defined
from utils_custom import calc_psnr, calc_ssim, calc_sssim # modify utils to utils_custom to avoid conflict with Python package 'utils'

# define image files
img_path_20042790 = "sr_data/lr_images_10/lr_20042790.fits"
img_path_20043513 = "sr_data/lr_images_10/lr_20043513.fits"
img_path_20160226 = "sr_data/lr_images_10/lr_20160226.fits"

img_path_20042790_hr = "sr_data/hr_images/20042790.fits"
img_path_20043513_hr = "sr_data/hr_images/20043513.fits"
img_path_20160226_hr = "sr_data/hr_images/20160226.fits"

img_20042790  = get_pkg_data_filename(img_path_20042790)
img_20042790 = fits.getdata(img_20042790, ext=0)
img_20042790 = img_20042790.astype(np.float32)
img_20042790 = img_20042790.reshape(1,256,256)

img_20042790_hr  = get_pkg_data_filename(img_path_20042790_hr)
img_20042790_hr = fits.getdata(img_20042790_hr, ext=0)
img_20042790_hr = img_20042790_hr.astype(np.float32)
img_20042790_hr = img_20042790_hr.reshape(1,256,256)

img_20043513  = get_pkg_data_filename(img_path_20043513)
img_20043513 = fits.getdata(img_20043513, ext=0)
img_20043513 = img_20043513.astype(np.float32)
img_20043513 = img_20043513.reshape(1,256,256)

img_20043513_hr  = get_pkg_data_filename(img_path_20043513_hr)
img_20043513_hr = fits.getdata(img_20043513_hr, ext=0)
img_20043513_hr = img_20043513_hr.astype(np.float32)
img_20043513_hr = img_20043513_hr.reshape(1,256,256)

img_20160226  = get_pkg_data_filename(img_path_20160226)
img_20160226 = fits.getdata(img_20160226, ext=0)
img_20160226 = img_20160226.astype(np.float32)
img_20160226 = img_20160226.reshape(1,256,256)

img_20160226_hr  = get_pkg_data_filename(img_path_20160226_hr)
img_20160226_hr = fits.getdata(img_20160226_hr, ext=0)
img_20160226_hr = img_20160226_hr.astype(np.float32)
img_20160226_hr = img_20160226_hr.reshape(1,256,256)

# define array for results
test_results_psnr_all = np.zeros((3,10))
test_results_ssim_all = np.zeros((3,10))
test_results_sssim_all = np.zeros((3,10))

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

outputs_dir = 'outputs/x3'

# img_20042790
for k_fold in range(10):
    # define and load model
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    save_path = f'./10_001-model-fold-{3}-{k_fold}.pth'

    for n, p in torch.load(os.path.join(outputs_dir, save_path), map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model.eval()

    inputs = img_20042790
    labels = img_20042790_hr
    inputs = torch.Tensor(inputs).to(device)
    labels = torch.Tensor(labels).to(device)
    
    with torch.no_grad():
        preds = model(inputs).clamp(0.0, 1.0)

    img_name = "img_results_10_2/test_20042790_" + str(k_fold) + ".fits"
    hdu = fits.PrimaryHDU(preds.detach().cpu().numpy())
    hdu.writeto(img_name, overwrite=True)
                
    test_results_psnr_all[0, k_fold] = calc_psnr(preds, labels)
    test_results_ssim_all[0, k_fold] = calc_ssim(preds, labels)
    test_results_sssim_all[0, k_fold] = calc_sssim(preds, labels)

# img_20043513
for k_fold in range(10):
    # define and load model
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    save_path = f'./10_001-model-fold-{2}-{k_fold}.pth'

    for n, p in torch.load(os.path.join(outputs_dir, save_path), map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model.eval()

    inputs = img_20043513
    labels = img_20043513_hr
    inputs = torch.Tensor(inputs).to(device)
    labels = torch.Tensor(labels).to(device)
    
    with torch.no_grad():
        preds = model(inputs).clamp(0.0, 1.0)

    img_name = "img_results_10_2/test_20043513_" + str(k_fold) + ".fits"
    hdu = fits.PrimaryHDU(preds.detach().cpu().numpy())
    hdu.writeto(img_name, overwrite=True)
                
    test_results_psnr_all[1, k_fold] = calc_psnr(preds, labels)
    test_results_ssim_all[1, k_fold] = calc_ssim(preds, labels)
    test_results_sssim_all[1, k_fold] = calc_sssim(preds, labels)


# img_20160226
for k_fold in range(10):
    # define and load model
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    save_path = f'./10_001-model-fold-{4}-{k_fold}.pth'

    for n, p in torch.load(os.path.join(outputs_dir, save_path), map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model.eval()

    inputs = img_20160226
    labels = img_20160226_hr
    inputs = torch.Tensor(inputs).to(device)
    labels = torch.Tensor(labels).to(device)
    
    with torch.no_grad():
        preds = model(inputs).clamp(0.0, 1.0)

    img_name = "img_results_10_2/test_20160226_" + str(k_fold) + ".fits"
    hdu = fits.PrimaryHDU(preds.detach().cpu().numpy())
    hdu.writeto(img_name, overwrite=True)
                
    test_results_psnr_all[2, k_fold] = calc_psnr(preds, labels)
    test_results_ssim_all[2, k_fold] = calc_ssim(preds, labels)
    test_results_sssim_all[2, k_fold] = calc_sssim(preds, labels)

# save csv files

test_psnr_all_name = "img_results_10_2/test_results_psnr_all.csv"
test_ssim_all_name = "img_results_10_2/test_results_ssim_all.csv"
test_sssim_all_name = "img_results_10_2/test_results_sssim_all.csv"

test_results_psnr_all_df = pd.DataFrame(test_results_psnr_all)
test_results_psnr_all_df.to_csv(test_psnr_all_name)
test_results_ssim_all_df = pd.DataFrame(test_results_ssim_all)
test_results_ssim_all_df.to_csv(test_ssim_all_name)
test_results_sssim_all_df = pd.DataFrame(test_results_sssim_all)
test_results_sssim_all_df.to_csv(test_sssim_all_name)
