import numpy as np
from osgeo import gdal
from scipy import fftpack
import cv2
from createTrainData import sar

num=0
#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48.txt"
#"E:\ALOSPALSAR\TrainData\ALPSRP180031440\ALPSRP180031440_SLC_24.txt"
RECT_SIZE=18
win_size=RECT_SIZE

hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))




SLCtxt = open("E:\\ALOSPALSAR\\TrainData\\SLC36.txt", 'r')
spetxt=open("E:\\ALOSPALSAR\\TrainData\\spe18.txt", 'w')

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return


while True:
    line = SLCtxt.readline()
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
            ff0 = spectrogram[K, :, :]
            ff += (ff0 - (ff0.min())) / ((ff0.max()) - (ff0.min()))
        ff = ff / (x * y)
        cv2.imwrite('E:\\ALOSPALSAR\\TrainData\\spe18'+'\\'+'spe_'+str(num)+'.tif',ff)
        #path+'\\'+sar+'\\'+sar+'_24\\'+sar+
        spetxt.write('E:\\ALOSPALSAR\\TrainData\\spe18'+'\\'+'spe_'+str(num)+'.tif'+' '+str(img_label))
        num=num+1
    else:
        break
SLCtxt.close()
spetxt.close()