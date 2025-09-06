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
    def __init__(self, csv_file, image_format):
        self.annotations = pd.read_csv(csv_file)
        self.image_format = image_format

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path_hr = self.annotations.iloc[index, 0]
        img_path_lr = self.annotations.iloc[index, 1]
        
        if self.image_format == 'fits':
            image_hr = fits.getdata(img_path_hr, ext=0).astype(np.float32)
            image_lr = fits.getdata(img_path_lr, ext=0).astype(np.float32)
        elif self.image_format == 'tif':
            image_hr = np.array(Image.open(img_path_hr)).astype(np.float32)
            image_lr = np.array(Image.open(img_path_lr)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported image format: {self.image_format}")
        
        return image_lr, image_hr
