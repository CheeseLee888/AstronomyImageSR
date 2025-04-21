# Code adapted from https://github.com/Fivefold/SRCNN/blob/main/Torch/util.py

import torch
import numpy as np

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch  
import torch.nn.functional as F 

#cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_sssim(img1, img2, window_size=11, k1=0.01, k2=0.03, sigma=1.5):

    shape_flag = 0

    if img1.shape != img2.shape:
        print("ValueError: differently sized input images")
        return None
    
    if window_size % 2 == 0:
        print("ValueError: window size is not odd")
        return None
        
    try:
        _, img_channels, height, width = img1.size()
    except:
        img_channels, height, width = img1.size()
        shape_flag = 1

    # get weighted array
    img_1_np = img1.cpu().numpy()
    img_2_np = img2.cpu().numpy()
    # create array that has brightness value for each pixel 
    data_1 = img_1_np.flatten()
    data_1 = img_1_np.reshape(-1,1)
    kmeans_1 = KMeans(n_clusters=3, n_init=10, random_state=23)
    kmeans_1.fit(data_1)
    df_1 = pd.DataFrame({'brightness': img_1_np.flatten(),
                       'label': kmeans_1.labels_})
    background_label_1 = df_1.loc[df_1['brightness'].idxmin()]['label']
    data_1 = data_1.reshape(img_1_np.shape)
    if shape_flag == 0:
        labels_1 = kmeans_1.labels_.reshape(img_1_np.shape[2],img_1_np.shape[3])
    else: 
        labels_1 = kmeans_1.labels_.reshape(img_1_np.shape[1],img_1_np.shape[2])
    
    weighted_array_1 = 1*(labels_1 != background_label_1)

    # create array that has brightness value for each pixel 
    data_2 = img_2_np.flatten()
    data_2 = img_2_np.reshape(-1,1)
    kmeans_2 = KMeans(n_clusters=3, n_init=10, random_state=23)
    kmeans_2.fit(data_2)
    df_2 = pd.DataFrame({'brightness': img_2_np.flatten(),
                       'label': kmeans_2.labels_})
    background_label_2 = df_2.loc[df_2['brightness'].idxmin()]['label']
    data_2 = data_2.reshape(img_2_np.shape)
    if shape_flag == 0:
        labels_2 = kmeans_2.labels_.reshape(img_2_np.shape[2],img_1_np.shape[3])
    else: 
        labels_2 = kmeans_2.labels_.reshape(img_2_np.shape[1],img_1_np.shape[2])
    
    weighted_array_2 = 1*(labels_2 != background_label_2)

    weighted_array = weighted_array_1 + weighted_array_2
    weighted_array = weighted_array.reshape(img_1_np.shape)
    weighted_array = np.where(weighted_array > 0, 1, 0)    
    weighted_array = torch.Tensor(weighted_array).to(device)

    # we set L to be the dynamic range of pixel values
    L = 1 #np.max(np.max(img_1_np)-np.min(img_1_np), np.max(img_2_np)-np.min(img_2_np))
    # we define the padding size for the sliding window
    pad = window_size // 2
    
    # initialize sliding window
    # 1d tensor
    window_1d = torch.Tensor(np.exp(-(np.arange(window_size) - window_size//2)**2/float(2*sigma**2)))
    window_1d /= window_1d.sum()
    window_1d = window_1d.unsqueeze(1)

    # convert to 2d
    window_2d = torch.Tensor(np.outer(window_1d, window_1d)).unsqueeze(0).unsqueeze(0).to(device)
    
    window = torch.Tensor(window_2d.expand(img_channels, 1, window_size, window_size).contiguous()).to(device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=img_channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=img_channels)

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=img_channels) - mu1**2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=img_channels) - mu2**2
    sigma_12 =  F.conv2d(img1 * img2, window, padding=pad, groups=img_channels) - mu1*mu2

    # Define stability constants
    C1 = k1 ** 2  
    C2 = k2 ** 2 

    num1 = 2 * mu1*mu2 + C1  
    num2 = 2 * sigma_12 + C2
    denom1 = mu1**2 + mu2**2 + C1 
    denom2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (num1 * num2) / (denom1 * denom2)    
    ssim_score = ssim_score.to(device)
    weighted_ssim_score = torch.mul(weighted_array,ssim_score)

    return torch.sum(weighted_ssim_score) / torch.sum(weighted_array)

def calc_ssim(img1, img2, window_size=11, k1=0.01, k2=0.03, sigma=1.5):
    
    if img1.shape != img2.shape:
        print("ValueError: differently sized input images")
        return None
    
    if window_size % 2 == 0:
        print("ValueError: window size is not odd")
        return None
        
    try:
        _, img_channels, height, width = img1.size()
    except:
        img_channels, height, width = img1.size()
    
    # we set L to be the dynamic range of pixel values
    L = 1 #np.max(np.max(img_1_np)-np.min(img_1_np), np.max(img_2_np)-np.min(img_2_np))
    # we define the padding size for the sliding window
    pad = window_size // 2
    
    # initialize sliding window
    # 1d tensor
    window_1d = torch.Tensor(np.exp(-(np.arange(window_size) - window_size//2)**2/float(2*sigma**2)))
    window_1d /= window_1d.sum()
    window_1d = window_1d.unsqueeze(1)

    img1 = img1.to(device)
    img2 = img2.to(device)

    # convert to 2d
    window_2d = torch.Tensor(np.outer(window_1d, window_1d)).unsqueeze(0).unsqueeze(0)
    
    window = torch.Tensor(window_2d.expand(img_channels, 1, window_size, window_size).contiguous()).to(device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=img_channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=img_channels)

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=img_channels) - mu1**2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=img_channels) - mu2**2
    sigma_12 =  F.conv2d(img1 * img2, window, padding=pad, groups=img_channels) - mu1*mu2

    # Define stability constants
    C1 = k1 ** 2  
    C2 = k2 ** 2 

    num1 = 2 * mu1*mu2 + C1  
    num2 = 2 * sigma_12 + C2
    denom1 = mu1**2 + mu2**2 + C1 
    denom2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (num1 * num2) / (denom1 * denom2)    
    
    return ssim_score.mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.var = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
