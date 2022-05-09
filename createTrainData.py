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

# E:\ALOSPALSAR\Greenland201101\510
ST=ALPSRP267211510

GCPdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2, 7))
XYdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2))
label = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(7))
XYsize = np.ones_like(GCPdata[:, 0:2])

RECT_SIZE = 12
# os.mkdir("1_size"+str(2*RECT_SIZE))
XYul = XYdata - RECT_SIZE * XYsize
XYdr = XYdata + RECT_SIZE * XYsize

# ALOS-P1_1__A-ORBIT__ALPSRP256411570_Cal_ML_Spk_Decomppauli
#"E:\ALOSPALSAR\Greenland201101\510\ALOS-P1_1__A-ORBIT__ALPSRP267211510_Cal_ML_Spk_Decomppauli.tif"
ds = gdal.Open(r'E:\ALOSPALSAR\Greenland201101\510\ALOS-P1_1__A-ORBIT__ALPSRP267211510_Cal_ML_Spk_Decomppauli.tif')
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据
im_data3 = band3.ReadAsArray()  # 获取数据

im_data1 = (im_data1 - (ST[0][0])) / ((ST[0][1]) - (ST[0][0]))
im_data2 = (im_data2 - (ST[1][0])) / ((ST[1][1]) - (ST[1][0]))
im_data3 = (im_data3 - (ST[2][0])) / ((ST[2][1]) - (ST[2][0]))
i = 2
cropped1 = im_data1[32:80, 131:179]
cropped2 = im_data2[XYul[0][0]:XYdr[0][0], XYul[0][1]:XYdr[0][1]]
cropped3 = im_data3[XYul[0][0]:XYdr[0][0], XYul[0][1]:XYdr[0][1]]

print(XYul, XYdr)
plt.imshow(cropped1)
plt.xticks([]), plt.yticks([])  # 不显示坐标轴
plt.show()


def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1, np2, np3):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    out_band = out_ds.GetRasterBand(2)
    out_band.WriteArray(np2)
    out_band = out_ds.GetRasterBand(3)
    out_band.WriteArray(np3)
    return

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510.txt"
labeltxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510.txt', 'w')
# print(tensor1)
#E:\ALOSPALSAR\TrainData\ALPSRP267211510
for i in range(0, len(GCPdata)):
    nnpp1 = im_data1[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp2 = im_data2[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp3 = im_data3[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]

    labelid=label[i]
    write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510' + '\\' + 'ALPSRP267211510_' + str(i) + '_0509_'+str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              3, band1.DataType, nnpp1, nnpp2, nnpp3)
    labeltxt.write('E:\ALOSPALSAR\TrainData\ALPSRP267211510\\' + 'ALPSRP267211510_' + str(i) + '_0509_'+str(labelid)+'.tif' + ' ' + str(labelid)+'\n')

labeltxt.close()

#ALPSRP267211510 r -39.0113639831543 -13.956258773803711  g -42.140995025634766 -15.663641929626465 b -36.93307876586914 -4.1313347816467285