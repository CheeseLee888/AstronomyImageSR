from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
# 打开 FITS 文件
hdul = fits.open("/Users/peterli/Desktop/BS_thesis/hide/nicmos_hr/nicmos_H_095746+0220_sci_25.fits")
image_data = hdul[0].data
hdul.close()

# 可视化
plt.imshow(np.log1p(image_data), cmap='gray')
plt.colorbar()
plt.title("FITS Image")
plt.show()

print(image_data.min(), image_data.max(), image_data.mean())
