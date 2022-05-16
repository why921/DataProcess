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
from img_statistics import ALPSRP267211510

GCPdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2, 7))
XYdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2))
label = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(7))
XYsize = np.ones_like(GCPdata[:, 0:2])

RECT_SIZE = 24
# os.mkdir("1_size"+str(2*RECT_SIZE))
XYul = XYdata - RECT_SIZE * XYsize
XYdr = XYdata + RECT_SIZE * XYsize
#"E:\ALOSPALSAR\Greenland201101\510\ALOS-P1_1__A-ORBIT__ALPSRP267211510_Cal_ML.tif"
ds = gdal.Open(r'E:\ALOSPALSAR\Greenland201101\510\ALOS-P1_1__A-ORBIT__ALPSRP267211510_Cal_ML.tif')

rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)

im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据

img=im_data1+1j*im_data2

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return

labeltxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510_SLC_48.txt', 'w')
# print(tensor1)
#E:\ALOSPALSAR\TrainData\ALPSRP267211510
#ALPSRP267211510_SLC_48
for i in range(0, len(GCPdata)):
    nnpp1 = img[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    labelid=label[i]
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510_SLC_48' + '\\' + 'ALPSRP267211510_SLC_' + str(i) + '_0509_'+str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              3, band1.DataType, nnpp1)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510_SLC_48\\' + 'ALPSRP267211510_SLC_' + str(i) + '_0509_'+str(labelid)+'.tif' + ' ' + str(labelid)+'\n')

labeltxt.close()