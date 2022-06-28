import numpy as np
from osgeo import gdal
from scipy import fftpack
#from ValidationData import sar
import cv2

sar='ALPSRP201761520'
num=0

win_size=24

hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands24.txt"
#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\slc48.txt"
imgtxt = open('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\slc'+str(2*win_size)+'.txt', 'r')
SLCtxt=open('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\spe4bands'+str(win_size)+'.txt', 'w')

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1,np2,np3,np4):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    out_band = out_ds.GetRasterBand(2)
    out_band.WriteArray(np2)
    out_band = out_ds.GetRasterBand(3)
    out_band.WriteArray(np3)
    out_band = out_ds.GetRasterBand(4)
    out_band.WriteArray(np4)
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
        band2 = ds.GetRasterBand(2)
        band3 = ds.GetRasterBand(3)
        band4 = ds.GetRasterBand(4)
        slc_data1 = band1.ReadAsArray()  # 获取数据
        slc_data2 = band2.ReadAsArray()  # 获取数据
        slc_data3 = band3.ReadAsArray()  # 获取数据
        slc_data4 = band4.ReadAsArray()  # 获取数据
        slc_fft1 = fftpack.fftshift(fftpack.fft2(slc_data1))
        slc_fft2 = fftpack.fftshift(fftpack.fft2(slc_data2))
        slc_fft3 = fftpack.fftshift(fftpack.fft2(slc_data3))
        slc_fft4 = fftpack.fftshift(fftpack.fft2(slc_data4))
        spectrogram1 = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
        spectrogram2 = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
        spectrogram3 = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
        spectrogram4 = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
        ff1 = np.zeros((win_size, win_size))
        ff2 = np.zeros((win_size, win_size))
        ff3 = np.zeros((win_size, win_size))
        ff4 = np.zeros((win_size, win_size))

        for i in range(0, win_size):
            for j in range(0, win_size):
                spectrogram1[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft1[i:i + win_size, j:j + win_size],
                                                         shape=[win_size, win_size])
                spectrogram2[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft2[i:i + win_size, j:j + win_size],
                                                         shape=[win_size, win_size])
                spectrogram3[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft3[i:i + win_size, j:j + win_size],
                                                         shape=[win_size, win_size])
                spectrogram4[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft4[i:i + win_size, j:j + win_size],
                                                         shape=[win_size, win_size])
        spectrogram1 = np.log(1 + np.abs(spectrogram1))
        spectrogram2 = np.log(1 + np.abs(spectrogram2))
        spectrogram3 = np.log(1 + np.abs(spectrogram3))
        spectrogram4 = np.log(1 + np.abs(spectrogram4))
        x, y, fr, fa = spectrogram1.shape
        spectrogram1 = spectrogram1.reshape([x * y, fr, fa])
        spectrogram2 = spectrogram2.reshape([x * y, fr, fa])
        spectrogram3 = spectrogram3.reshape([x * y, fr, fa])
        spectrogram4 = spectrogram4.reshape([x * y, fr, fa])
        for K in range(0, (x * y)):
            ff1 += spectrogram1[K, :, :]
            ff2 += spectrogram2[K, :, :]
            ff3 += spectrogram3[K, :, :]
            ff4 += spectrogram4[K, :, :]
        ff1 = ff1 / (x * y)
        ff2 = ff2 / (x * y)
        ff3 = ff3 / (x * y)
        ff4 = ff4 / (x * y)
        #E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands24
       
        write_img('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\spe4bands'+str(win_size)+'\\'+'spe4bands_'+str(num)+'.tif',win_size,win_size,4,band1.DataType,ff1,ff2,ff3,ff4)

        SLCtxt.write('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\spe4bands'+str(win_size)+'\spe4bands_'+str(num)+'.tif'+' '+str(img_label))
        num=num+1
    else:
        break
imgtxt.close()
SLCtxt.close()