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
from scipy import fftpack
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
#Joint time-frequency analysis for radar signal and imaging
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_0_0509_0.tif"
#A NEW IMAGING METHOD FOR QUASI GEOSTATIONARY SAR CONSTELLATION USING SPECTRUM GAP FILLING
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_65_0509_1.tif 1   57 58 59 60
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_80_0509_2.tif 79-90
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_54_0509_3.tif
#E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_53_0509_3.tif
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48\ALPSRP267211510_SLC_HH_0_0509_0.tif"
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510_SLC_48\ALPSRP267211510_SLC_0_0509_0.tif"
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48\ALPSRP267211510_SLC_0_0517.tif"
ds = gdal.Open(r'E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48\ALPSRP267211510_SLC_HH_0_0509_0.tif')
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
ds2 = gdal.Open(r'E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48\ALPSRP267211510_SLC_0_0517.tif')
band2=ds2.GetRasterBand(1)
im_data1 = band1.ReadAsArray()  # 获取数据
im_data2=band2.ReadAsArray()

f = np.fft.fft2(im_data1)

# 默认结果中心点位置是在左上角,
# 调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

# fft结果是复数, 其绝对值结果是振幅
fimg = np.log(np.abs(fshift))

win_size=24
hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
slc_fft = fftpack.fftshift(fftpack.fft2(im_data1))
for i in range(0,win_size):
    for j in range(0,win_size):
        spectrogram[:,:,i,j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i+win_size, j:j+win_size],
                                             shape=[win_size, win_size])
spectrogram = np.log(1+np.abs(spectrogram))
x, y, fr, fa = spectrogram.shape
spectrogram = spectrogram.reshape([x*y, fr, fa])
spectrogramxfa = spectrogram.reshape([x, y*fr, fa])
fxfa1=spectrogramxfa[:,100,:]
f1=spectrogram[55,:,:]
f2=spectrogram[1,:,:]
f3=spectrogram[100,:,:]
ff=np.zeros((24,24))
a=0
for i in range(0, (x*y)):
    ff+=spectrogram[i,:,:]
    a=a+1

ff=ff/(x*y)
# 展示结果
print(a)
print(x*y)
plt.subplot(331), plt.imshow(im_data2, 'gray'), plt.title('Original Fourier')
plt.axis('off')
plt.subplot(332), plt.imshow(f1, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.subplot(333), plt.imshow(f2, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.subplot(334), plt.imshow(f3, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.subplot(335), plt.imshow(ff, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.subplot(336), plt.imshow(fxfa1, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.show()