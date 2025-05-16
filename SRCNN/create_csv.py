# File to create CSV file with LR and HR image locations needed for the creation of the data set

import glob
import pandas as pd
import os

"""
THE ONLY CELL BELOW: MODIFY VALUES TO CATER
"""

image_dataset_name = 'Hubble_Images_top90_256x256' # image dataset folder name
image_format = 'tif' # 'tif' or 'fits'
change = '4_001' #  '4_001' or '10_001' or '10_0015'



astropy_images = pd.DataFrame()
name_list_hr = []
name_list_lr = []
ind=0

se_hr_dir = '../../Prep_Images/'+image_dataset_name+'_se_hr_'+change # input: selective HR and LR images from preparation step
se_lr_dir = '../../Prep_Images/'+image_dataset_name+'_se_lr_'+change

for filename in glob.glob(os.path.join(se_hr_dir, '*.'+image_format)):
    filename_hr = filename
    filename_lr = os.path.join(se_lr_dir, 'lr_' + filename.split("/")[-1])
    name_list_hr.append(filename_hr)
    name_list_lr.append(filename_lr)


astropy_images['HR'] = name_list_hr
astropy_images['LR'] = name_list_lr

os.makedirs('srcnn_prep_filename', exist_ok=True)
astropy_images.to_csv('srcnn_prep_filename/'+image_dataset_name+'_'+change+'.csv', index=False)
