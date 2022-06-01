import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from osgeo import gdal
from scipy import fftpack
import cv2
import torch
import DataTrans
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


win_size=12
num=0
hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
#E:\DataSetProcess\gtxt.txt
gtxt=open('E:\DataSetProcess\gtxt.txt', 'w')
def show(img):
    plt.figure(figsize=(12, 12))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return
#"E:\ALOSPALSAR\imgshow\Beaufort91510SLC\ALPSRP267211510_SLC_HH_0_0509_2.tif"
#"E:\ALOSPALSAR\imgshow\Beaufort51550SLC\ALPSRP267211510_SLC_HH_0_0509_2.tif"
ds = gdal.Open('E:\ALOSPALSAR\imgshow\Beaufort51550SLC\ALPSRP267211510_SLC_HH_5_0509_0.tif')
band1 = ds.GetRasterBand(1)
slc_data1 = band1.ReadAsArray()

slc_fft = fftpack.fftshift(fftpack.fft2(slc_data1))
spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
ff = np.zeros((win_size, win_size))
ss= np.zeros((win_size, win_size))

for i in range(0, win_size):
    for j in range(0, win_size):
        spectrogram[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i + win_size, j:j + win_size],
                                                        shape=[win_size, win_size])
spectrogram = np.log(1 + np.abs(spectrogram))
x, y, fr, fa = spectrogram.shape
spectrogram = spectrogram.reshape([x * y, fr, fa])
for K in range(0, (x * y)):
    ff = spectrogram[K, :, :]
    ff=(ff - (ff.min())) / ((ff.max()) - (ff.min()))
    ss+=ff
    #E:\ALOSPALSAR\imgshow\temp
    num=num+1
ss=ss/144
cv2.imwrite('AAA.tif', ss)
gtxt.close()