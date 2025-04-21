# File to create CSV file with LR and HR image locations needed for the creation of the data set

import glob
import pandas as pd
import os

### 
#唯一需要修改的地方！！！
###
change='4_001' # IMPORTANT: choose the change from hr to lr here, '10_0015' or '10_001' or '4_001'



astropy_images = pd.DataFrame()
name_list_hr = []
name_list_lr = []
ind=0

se_hr_dir = "/Users/peterli/Desktop/BS_thesis/nicmos_se_hr_"+change
se_lr_dir = "/Users/peterli/Desktop/BS_thesis/nicmos_se_lr_"+change

for filename in glob.glob(os.path.join(se_hr_dir, "*.fits")):
    # filename = filename.split("/")[-1]
    filename_hr = filename
    filename_lr = os.path.join(se_lr_dir, 'lr_' + filename.split("/")[-1])
    name_list_hr.append(filename_hr)
    name_list_lr.append(filename_lr)

# # 原始的代码
# for filename in glob.glob('/home/julia/SRCNN/sr_data/hr_images/*.fits'):
#     filename = filename.split("/")[-1]
#     filename_hr = 'sr_data/hr_images/' + filename
#     filename_lr = 'sr_data/lr_images_10_noise_0015/lr_' + filename
#     name_list_hr.append(filename_hr)
#     name_list_lr.append(filename_lr)


astropy_images['HR'] = name_list_hr
astropy_images['LR'] = name_list_lr
astropy_images.to_csv('srcnn_prep_images_'+change+'.csv', index=False)
