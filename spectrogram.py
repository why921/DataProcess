import numpy as np
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
num=0
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48.txt"
win_size=24

hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))

#ALPSRP267211510_spectrogram_48
imgtxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48.txt', 'r')
SLCtxt=open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48.txt', 'w')

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return
#write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48' + '\\' + 'ALPSRP267211510_SLC_VV_' + str(i) + '_0509_' + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
        #      3, band1.DataType, nnpp4)

while True:
    line = imgtxt.readline()
    if line:
        img_path=line.split(" ")[0]
        img_label=line.split(" ")[1]
        ds = gdal.Open(str(img_path))
        band1 = ds.GetRasterBand(1)
        slc_data1 = band1.ReadAsArray()  # 获取数据
        slc_fft = fftpack.fftshift(fftpack.fft2(slc_data1))
        spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
        ff = np.zeros((win_size, win_size))
        for i in range(0, win_size):
            for j in range(0, win_size):
                spectrogram[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i + win_size, j:j + win_size],
                                                        shape=[win_size, win_size])
        spectrogram = np.log(1 + np.abs(spectrogram))
        x, y, fr, fa = spectrogram.shape
        spectrogram = spectrogram.reshape([x * y, fr, fa])
        for K in range(0, (x * y)):
            ff += spectrogram[K, :, :]
        ff = ff / (x * y)
        cv2.imwrite('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48'+'\\'+'ALPSRP267211510_SLC_'+str(num)+'_0517.tif',ff)
        SLCtxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48\ALPSRP267211510_SLC_'+str(num)+'_0517.tif'+' '+str(img_label))
        num=num+1
    else:
        break
imgtxt.close()
SLCtxt.close()