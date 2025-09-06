"""
CALCULATE PSNR, SSIM, SSSIM BETWEEN (SELECTIVE) HR AND LR IMAGES IN ALL FORMATS
"""
import astropy
from astropy.io import fits
from PIL import Image
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel, convolve
from astropy import units as u
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np 
import glob
import os
import matplotlib.pyplot as plt
import torch

from utils_custom import calc_psnr, calc_ssim, calc_sssim # modify utils to utils_custom to avoid conflict with Python package 'utils'

ssim_vals = []
sssim_vals = []
psnr_vals = []

image_format = 'tif' # 'fits' or 'tif'
img_dir = '../compare'
filename_1 = 'hr.tif'
filename_2 = 'srcnn_4_001_k=2.tif'

if image_format == 'fits':
    img_1 = get_pkg_data_filename(os.path.join(img_dir, filename_1))
    imarray_1 = fits.getdata(img_1, ext=0).astype(np.float32)
    img_2 = get_pkg_data_filename(os.path.join(img_dir, filename_2))
    imarray_2 = fits.getdata(img_2, ext=0).astype(np.float32)
else: # tif
    img_1 = Image.open((os.path.join(img_dir, filename_1)))
    imarray_1 = np.array(img_1).astype(np.float32)
    img_2 = Image.open((os.path.join(img_dir, filename_2)))
    imarray_2 = np.array(img_2).astype(np.float32)


img_1 = imarray_1.reshape(1, imarray_1.shape[0], imarray_1.shape[1])
img_2 = imarray_2.reshape(1, imarray_2.shape[0], imarray_2.shape[1])

ssim_vals = calc_ssim(torch.Tensor(img_1), torch.Tensor(img_2)).item()
sssim_vals = calc_sssim(torch.Tensor(img_1), torch.Tensor(img_2)).item()

print(filename_1, ' vs ', filename_2)
print('SSIM: ', ssim_vals)
print('SSSIM: ', sssim_vals)