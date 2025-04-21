import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


# Selective SSIM / SSIM implementation adapted from Benjamin Kan's implementation on
#  https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow, accessed on 08/08/2023

def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def sssim_single_image(img1, img2, size=11, sigma=1.5):

    # get weighted array
    img_1_np = img1#.cpu().numpy()
    img_2_np = img2#.cpu().numpy()
    # create array that has brightness value for each pixel 
    data_1 = img_1_np.flatten()
    data_1 = img_1_np.reshape(-1,1)
    data_2 = img_2_np.flatten()
    data_2 = img_2_np.reshape(-1,1)
    kmeans_1 = KMeans(n_clusters=3, n_init=10, random_state=23)
    kmeans_2 = KMeans(n_clusters=3, n_init=10, random_state=23)
    kmeans_1.fit(data_1)
    kmeans_2.fit(data_2)
    df_1 = pd.DataFrame({'brightness': img_1_np.flatten(),
                       'label': kmeans_1.labels_})
    df_2 = pd.DataFrame({'brightness': img_2_np.flatten(),
                       'label': kmeans_2.labels_})
    background_label_1 = df_1.loc[df_1['brightness'].idxmin()]['label']
    background_label_2 = df_2.loc[df_2['brightness'].idxmin()]['label']
    data_1 = data_1.reshape(img_1_np.shape)
    labels_1 = kmeans_1.labels_.reshape(img_1_np.shape[0],img_1_np.shape[1])

    data_2 = data_2.reshape(img_2_np.shape)
    labels_2 = kmeans_2.labels_.reshape(img_2_np.shape[0],img_2_np.shape[1])
    detailed_array_1 = 1*(labels_1 != background_label_1)
    detailed_array_2 = 1*(labels_2 != background_label_2)
    weighted_array = (detailed_array_1.reshape(img_1_np.shape) +  detailed_array_2.reshape(img_2_np.shape))
    weighted_array = np.where(weighted_array > 0, 1, 0)
    total_pixels = np.sum(weighted_array)
    total_pixels = tf.constant(total_pixels, dtype=tf.float32)
    weighted_array =  tf.convert_to_tensor(weighted_array, np.float32)
    img1 = image_to_4d(img1)
    img2 = image_to_4d(img2)
    weighted_array = image_to_4d(weighted_array)

    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
            (sigma1_sq + sigma2_sq + C2))

    value = tf.math.multiply(value, weighted_array)
    a = tf.reduce_sum(value)
    value = tf.divide(a, total_pixels)
    return value 
    
def keras_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1)

def _tf_crop(im, crop=320):
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    im = im[:, starty:starty+crop, startx:startx+crop, :]
    return im

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

def center_keras_psnr(y_true, y_pred):
    return tf.image.psnr(_tf_crop(y_true, crop=128), _tf_crop(y_pred, crop=128), 1)

def keras_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1)

def psnr_single_image(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=1)

def ssim_single_image(img1, img2, size=11, sigma=1.5):
    img1 = image_to_4d(img1)
    img2 = image_to_4d(img2)
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
            (sigma1_sq + sigma2_sq + C2))
    result = tf.reduce_mean(value)
    return result 

def psnr(gts, preds):
    """Compute the psnr of a batch of images in HWC format.

    Images must be in NHWC format
    """
    if len(gts.shape) == 3:
        return psnr_single_image(gts, preds)
    else:
        mean_psnr = [psnr_single_image(gt, pred) for gt, pred in zip(gts, preds)]
        return mean_psnr

def ssim(gts, preds):
    """Compute the ssim of a batch of images in HWC format.

    Images must be in NHWC format
    """
    if len(gts.shape) == 3:
        return ssim_single_image(gts, preds)
    else:
        mean_ssim = [ssim_single_image(gt, pred).numpy() for gt, pred in zip(gts, preds)]
        return mean_ssim

def sssim(gts, preds):
    """Compute the ssim of a batch of images in HWC format.

    Images must be in NHWC format
    """
    if len(gts.shape) == 3:
        return sssim_single_image(gts, preds)
    else:
        mean_sssim = [sssim_single_image(gt, pred).numpy() for gt, pred in zip(gts, preds)]
        return mean_sssim

METRIC_FUNCS = dict(
    PSNR=psnr,
    SSIM=ssim,
    SSSIM=sssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self):
        self.metrics = {
            metric: Statistics() for metric in METRIC_FUNCS
        }

    def push(self, target, recons, im_shape=None):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )