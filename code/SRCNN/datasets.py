import h5py
import numpy as np
import fits2hdf
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 1.,0), np.expand_dims(f['hr'][str(idx)][:, :] / 1.,0)

    #def __getitem__(self, idx):
    #    with h5py.File(self.h5_file, 'r') as f:
    #    return np.expand_dims(f['lr'][idx] / 1., 0), np.expand_dims(f['hr'][idx] / 1., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 1.,0), np.expand_dims(f['hr'][str(idx)][:, :] / 1.,0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
