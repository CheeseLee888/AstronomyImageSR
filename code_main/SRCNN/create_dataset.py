# File to define FITS data set to feed to SRCNN network 

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename 
from torch.utils.data import (
    Dataset,
) 

class AstropyDataset(Dataset):
    def __init__(self, csv_file):
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path_hr = self.annotations.iloc[index, 0]
        img_path_lr = self.annotations.iloc[index, 1]

        image_hr = fits.getdata(img_path_hr, ext=0).astype(np.float32)
        # image_hr  = image_hr.astype(np.float32)

        image_lr = fits.getdata(img_path_lr, ext=0).astype(np.float32)
        # image_lr  = image_lr.astype(np.float32)

        return image_lr, image_hr
