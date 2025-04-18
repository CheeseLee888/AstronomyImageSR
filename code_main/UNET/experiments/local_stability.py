import os
import pandas as pd
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
import pandas as pd


############### DATA SET 10.001


# load SRCNN images (Data Set 10.001)

img_path_0 = "img_results/20043513_2_0.fits"
img_path_1 = "img_results/20043513_2_1.fits"
img_path_2 = "img_results/20043513_2_2.fits"
img_path_3 = "img_results/20043513_2_3.fits"
img_path_4 = "img_results/20043513_2_4.fits"
img_path_5 = "img_results/20043513_2_5.fits"
img_path_6 = "img_results/20043513_2_6.fits"
img_path_7 = "img_results/20043513_2_7.fits"
img_path_8 = "img_results/20043513_2_8.fits"
img_path_9 = "img_results/20043513_2_9.fits"

img_0  = get_pkg_data_filename(img_path_0)
img_0 = fits.getdata(img_0, ext=0)
img_0 = img_0.astype(np.float32)
#img_0 = img_0.reshape(1,256,256)

img_1  = get_pkg_data_filename(img_path_1)
img_1 = fits.getdata(img_1, ext=0)
img_1 = img_1.astype(np.float32)
#img_1 = img_1.reshape(1,256,256)

img_2  = get_pkg_data_filename(img_path_2)
img_2 = fits.getdata(img_2, ext=0)
img_2 = img_2.astype(np.float32)
#img_2 = img_2.reshape(1,256,256)

img_3  = get_pkg_data_filename(img_path_3)
img_3 = fits.getdata(img_3, ext=0)
img_3 = img_3.astype(np.float32)
#img_3 = img_3.reshape(1,256,256)

img_4  = get_pkg_data_filename(img_path_4)
img_4 = fits.getdata(img_4, ext=0)
img_4 = img_4.astype(np.float32)
#img_4 = img_4.reshape(1,256,256)

img_5  = get_pkg_data_filename(img_path_5)
img_5 = fits.getdata(img_5, ext=0)
img_5 = img_5.astype(np.float32)
#img_5 = img_5.reshape(1,256,256)

img_6  = get_pkg_data_filename(img_path_6)
img_6 = fits.getdata(img_6, ext=0)
img_6 = img_6.astype(np.float32)
#img_6 = img_6.reshape(1,256,256)

img_7  = get_pkg_data_filename(img_path_7)
img_7 = fits.getdata(img_7, ext=0)
img_7 = img_7.astype(np.float32)
#img_7 = img_7.reshape(1,256,256)

img_8  = get_pkg_data_filename(img_path_8)
img_8 = fits.getdata(img_8, ext=0)
img_8 = img_8.astype(np.float32)
#img_8 = img_8.reshape(1,256,256)

img_9  = get_pkg_data_filename(img_path_9)
img_9 = fits.getdata(img_9, ext=0)
img_9 = img_9.astype(np.float32)
#img_9 = img_9.reshape(1,256,256)

#### get weighted array

# img 0
data_0 = img_0.flatten()
data_0 = img_0.reshape(-1,1)
kmeans_0 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_0.fit(data_0)
df_0 = pd.DataFrame({'brightness': img_0.flatten(),
                    'label': kmeans_0.labels_})
background_label_0 = df_0.loc[df_0['brightness'].idxmin()]['label']
data_0 = data_0.reshape(img_0.shape)
labels_0 = kmeans_0.labels_.reshape(img_0.shape[1],img_0.shape[2])
weighted_array_0 = 1*(labels_0 != background_label_0)

# img 1
data_1 = img_1.flatten()
data_1 = img_1.reshape(-1,1)
kmeans_1 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_1.fit(data_1)
df_1 = pd.DataFrame({'brightness': img_1.flatten(),
                    'label': kmeans_1.labels_})
background_label_1 = df_1.loc[df_1['brightness'].idxmin()]['label']
data_1 = data_1.reshape(img_1.shape)
labels_1 = kmeans_1.labels_.reshape(img_1.shape[1],img_1.shape[2])
    
weighted_array_1 = 1*(labels_1 != background_label_1)

# img 2
data_2 = img_2.flatten()
data_2 = img_2.reshape(-1,1)
kmeans_2 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_2.fit(data_2)
df_2 = pd.DataFrame({'brightness': img_2.flatten(),
                    'label': kmeans_2.labels_})
background_label_2 = df_2.loc[df_2['brightness'].idxmin()]['label']
data_2 = data_2.reshape(img_2.shape)
labels_2 = kmeans_2.labels_.reshape(img_2.shape[1],img_2.shape[2])
weighted_array_2 = 1*(labels_2 != background_label_2)

# img 3
data_3 = img_3.flatten()
data_3 = img_3.reshape(-1,1)
kmeans_3 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_3.fit(data_3)
df_3 = pd.DataFrame({'brightness': img_3.flatten(),
                    'label': kmeans_3.labels_})
background_label_3 = df_3.loc[df_3['brightness'].idxmin()]['label']
data_3 = data_3.reshape(img_3.shape)
labels_3 = kmeans_3.labels_.reshape(img_3.shape[1],img_3.shape[2])
weighted_array_3 = 1*(labels_3 != background_label_3)

# img 4
data_4 = img_4.flatten()
data_4 = img_4.reshape(-1,1)
kmeans_4 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_4.fit(data_4)
df_4 = pd.DataFrame({'brightness': img_4.flatten(),
                    'label': kmeans_4.labels_})
background_label_4 = df_4.loc[df_4['brightness'].idxmin()]['label']
data_4 = data_4.reshape(img_4.shape)
labels_4 = kmeans_4.labels_.reshape(img_4.shape[1],img_4.shape[2])
weighted_array_4 = 1*(labels_4 != background_label_4)

# img 5
data_5 = img_5.flatten()
data_5 = img_5.reshape(-1,1)
kmeans_5 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_5.fit(data_5)
df_5 = pd.DataFrame({'brightness': img_5.flatten(),
                    'label': kmeans_5.labels_})
background_label_5 = df_5.loc[df_5['brightness'].idxmin()]['label']
data_5 = data_5.reshape(img_5.shape)
labels_5 = kmeans_5.labels_.reshape(img_5.shape[1],img_5.shape[2])
weighted_array_5 = 1*(labels_5 != background_label_5)

# img 6
data_6 = img_6.flatten()
data_6 = img_6.reshape(-1,1)
kmeans_6 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_6.fit(data_6)
df_6 = pd.DataFrame({'brightness': img_6.flatten(),
                    'label': kmeans_6.labels_})
background_label_6 = df_6.loc[df_6['brightness'].idxmin()]['label']
data_6 = data_6.reshape(img_6.shape)
labels_6 = kmeans_6.labels_.reshape(img_6.shape[1],img_6.shape[2])
weighted_array_6 = 1*(labels_6 != background_label_6)

# img 7
data_7 = img_7.flatten()
data_7 = img_7.reshape(-1,1)
kmeans_7 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_7.fit(data_7)
df_7 = pd.DataFrame({'brightness': img_7.flatten(),
                    'label': kmeans_7.labels_})
background_label_7 = df_7.loc[df_7['brightness'].idxmin()]['label']
data_7 = data_7.reshape(img_7.shape)
labels_7 = kmeans_7.labels_.reshape(img_7.shape[1],img_7.shape[2])
weighted_array_7 = 1*(labels_7 != background_label_7)

# img 8
data_8 = img_8.flatten()
data_8 = img_8.reshape(-1,1)
kmeans_8 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_8.fit(data_8)
df_8 = pd.DataFrame({'brightness': img_8.flatten(),
                    'label': kmeans_8.labels_})
background_label_8 = df_8.loc[df_8['brightness'].idxmin()]['label']
data_8 = data_8.reshape(img_8.shape)
labels_8= kmeans_8.labels_.reshape(img_8.shape[1],img_8.shape[2])
weighted_array_8 = 1*(labels_8 != background_label_8)

# img 9
data_9 = img_9.flatten()
data_9 = img_9.reshape(-1,1)
kmeans_9 = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans_9.fit(data_9)
df_9 = pd.DataFrame({'brightness': img_9.flatten(),
                    'label': kmeans_9.labels_})
background_label_9 = df_9.loc[df_9['brightness'].idxmin()]['label']
data_9 = data_9.reshape(img_9.shape)
labels_9 = kmeans_9.labels_.reshape(img_9.shape[1],img_9.shape[2])
weighted_array_9 = 1*(labels_9 != background_label_9)

weighted_array = weighted_array_0 + weighted_array_1 + weighted_array_2 + weighted_array_3 + weighted_array_4 + weighted_array_5 + weighted_array_6 + weighted_array_7 + weighted_array_8 + weighted_array_9
weighted_array = weighted_array.reshape(img_0.shape)
weighted_array = np.where(weighted_array > 0, 1, 0)    

summed_imgs_0 = abs(img_0-img_1) + abs(img_0-img_2) + abs(img_0-img_3) + abs(img_0-img_4) + abs(img_0-img_5) + abs(img_0-img_6) + abs(img_0-img_7) + abs(img_0-img_8) + abs(img_0-img_9)
summed_imgs_1 = abs(img_1-img_2) + abs(img_1-img_3) + abs(img_1-img_4) + abs(img_1-img_5) + abs(img_1-img_6) + abs(img_1-img_7) + abs(img_1-img_8) + abs(img_1-img_9)
summed_imgs_2 = abs(img_2-img_3) + abs(img_2-img_4) + abs(img_2-img_5) + abs(img_2-img_6) + abs(img_2-img_7) + abs(img_2-img_8) + abs(img_2-img_9)
summed_imgs_3 = abs(img_3-img_4) + abs(img_3-img_5) + abs(img_3-img_6) + abs(img_3-img_7) + abs(img_3-img_8) + abs(img_3-img_9)
summed_imgs_4 = abs(img_4-img_5) + abs(img_4-img_6) + abs(img_4-img_7) + abs(img_4-img_8) + abs(img_4-img_9)
summed_imgs_5 = abs(img_5-img_6) + abs(img_5-img_7) + abs(img_5-img_8) + abs(img_5-img_9)
summed_imgs_6 = abs(img_6-img_7) + abs(img_6-img_8) + abs(img_6-img_9)
summed_imgs_7 = abs(img_7-img_8) + abs(img_7-img_9)
summed_imgs_8 = abs(img_8-img_9)

summed_imgs = summed_imgs_0 + summed_imgs_1 + summed_imgs_2 + summed_imgs_3 + summed_imgs_4 + summed_imgs_5 + summed_imgs_6 + summed_imgs_7 + summed_imgs_8 
summed_imgs = weighted_array * (1-summed_imgs/45)

masked_img = np.ma.masked_where(weighted_array == 0, (1-summed_imgs/45))

plt.figure()
plt.imshow(masked_img.reshape(256,256), cmap='PRGn')
cb1 = plt.colorbar()
for t in cb1.ax.get_yticklabels():
     t.set_fontsize(15)
#plt.clim(0.9,1.0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("heatmap/4_001/20043513_heatmap.png")
plt.close()

# detect background of HR image

hr_20160226 = "hr_images/20043513.fits"

hr_20160226_img  = get_pkg_data_filename(hr_20160226)
hr_20160226_img = fits.getdata(hr_20160226_img, ext=0)
hr_20160226_img = hr_20160226_img.astype(np.float32)
hr_20160226_img = hr_20160226_img.reshape(256,256)

data = hr_20160226_img.flatten()
data = hr_20160226_img.reshape(-1,1)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=23)
kmeans.fit(data)
df = pd.DataFrame({'brightness': hr_20160226_img.flatten(),
                    'label': kmeans.labels_})
background_label= df.loc[df['brightness'].idxmin()]['label']
data = data.reshape(hr_20160226_img.shape)

labels = kmeans.labels_.reshape(hr_20160226_img.shape[0],hr_20160226_img.shape[1])
    
weighted_array_hr = 1*(labels!= background_label)

weighted_differences = weighted_array_hr - weighted_array

masked_img_weighted = np.ma.masked_where(weighted_differences == 0, weighted_differences)

plt.figure()
plt.imshow(masked_img_weighted.reshape(256,256), cmap='PRGn')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("heatmap/4_001/weighted_array_differences_20043513.png")
plt.close()

masked_img_hr_weighted = np.ma.masked_where(weighted_array_hr == 0, weighted_array_hr)

plt.figure()
plt.imshow(masked_img_hr_weighted.reshape(256,256), cmap='PRGn')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("heatmap/4_001/weighted_array_hr_20043513.png")
plt.close()

masked_img_sr_weighted = np.ma.masked_where(weighted_array == 0, weighted_array)

plt.figure()
plt.imshow(masked_img_sr_weighted.reshape(256,256), cmap='PRGn')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("heatmap/4_001/weighted_array_sr_20043513.png")
plt.close()


