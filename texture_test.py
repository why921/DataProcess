import numpy as np
import numpy as np
import os
import torch
from osgeo import gdal
from skimage import data
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation

import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_0_0509_0.tif"
#A NEW IMAGING METHOD FOR QUASI GEOSTATIONARY SAR CONSTELLATION USING SPECTRUM GAP FILLING
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_65_0509_1.tif 1
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_80_0509_2.tif
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_54_0509_3.tif
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_53_0509_3.tif
ds = gdal.Open(r'E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_53_0509_3.tif')
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据
im_data3 = band3.ReadAsArray()  # 获取数据

f = np.fft.fft2(im_data1)

# 默认结果中心点位置是在左上角,
# 调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

# fft结果是复数, 其绝对值结果是振幅
fimg = np.log(np.abs(fshift))

# 展示结果
plt.subplot(121), plt.imshow(im_data1, 'gray'), plt.title('Original Fourier')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.show()