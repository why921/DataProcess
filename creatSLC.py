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

RECT_SIZE = 12
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
band3 = ds.GetRasterBand(3)
band4 = ds.GetRasterBand(4)
band5 = ds.GetRasterBand(5)
band6 = ds.GetRasterBand(6)
band7 = ds.GetRasterBand(7)
band8 = ds.GetRasterBand(8)

im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据
im_data3 = band3.ReadAsArray()  # 获取数据
im_data4 = band4.ReadAsArray()  # 获取数据
im_data5 = band5.ReadAsArray()  # 获取数据
im_data6 = band6.ReadAsArray()  # 获取数据
im_data7 = band7.ReadAsArray()  # 获取数据
im_data8 = band8.ReadAsArray()  # 获取数据

imgHH=im_data1+1j*im_data2
imgHV=im_data3+1j*im_data4
imgVH=im_data5+1j*im_data6
imgVV=im_data7+1j*im_data8

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return

labeltxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24.txt', 'w')
# print(tensor1)
#E:\ALOSPALSAR\TrainData\ALPSRP267211510
#ALPSRP267211510_SLC_48
for i in range(0, len(GCPdata)):
    nnpp1 = imgHH[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp2 = imgHV[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp3 = imgVH[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp4 = imgVV[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    labelid=label[i]
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24' + '\\' + 'ALPSRP267211510_SLC_HH_' + str(i) + '_0509_'+str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp1)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24\\' + 'ALPSRP267211510_SLC_HH_' + str(i) + '_0509_'+str(labelid)+'.tif' + ' ' + str(labelid)+'\n')
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24' + '\\' + 'ALPSRP267211510_SLC_HV_' + str(i) + '_0509_'+str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp2)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24\\' + 'ALPSRP267211510_SLC_HV_' + str(i) + '_0509_'+str(labelid)+'.tif' + ' ' + str(labelid)+'\n')
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24' + '\\' + 'ALPSRP267211510_SLC_VH_' + str(i) + '_0509_' + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp3)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24\\' + 'ALPSRP267211510_SLC_VH_' + str(i) + '_0509_' + str(labelid) + '.tif' + ' ' + str(labelid) + '\n')
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24' + '\\' + 'ALPSRP267211510_SLC_VV_' + str(i) + '_0509_' + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp4)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24\\' + 'ALPSRP267211510_SLC_VV_' + str(i) + '_0509_' + str(labelid) + '.tif' + ' ' + str(labelid) + '\n')

labeltxt.close()