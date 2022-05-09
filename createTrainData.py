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

#E:\ALOSPALSAR\Greenland201101\510

GCPdata = np.loadtxt('green_E_5_3.txt', dtype=int, skiprows=6, usecols=(1, 2))
XYsize = np.ones_like(GCPdata)

# ArrayToMat = np.mat(GCPdata)
# ArrayToMat = np.mat(XYsize)

RECT_SIZE=12
#os.mkdir("1_size"+str(2*RECT_SIZE))
XYul = GCPdata - RECT_SIZE * XYsize
XYdr = GCPdata + RECT_SIZE * XYsize
ArrayToMat = np.mat(XYul)
ArrayToMat = np.mat(XYdr)


print(XYul[0][0],XYdr[0][0], XYul[0][1],XYdr[0][1])
print(XYul[4][0],XYdr[4][0], XYul[4][1],XYdr[4][1])
# print(data)
# "D:\why2022\seaice\data_process\DC_subset_tif.tif"
#ALOS-P1_1__A-ORBIT__ALPSRP256411570_Cal_ML_Spk_Decomppauli
ds = gdal.Open(r'ALOS-P1_1__A-ORBIT__ALPSRP256411570_Cal_ML_Spk_Decomppauli.tif')
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)
#band4 = ds.GetRasterBand(4)
im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据
im_data3 = band3.ReadAsArray()  # 获取数据
#im_data4 = band4.ReadAsArray()  # 获取数据
#[np.array(XYul)[1]:np.array(XYdr)[1], np.array(XYul)[2]:np.array(XYdr)[2]]
i=2
# cv2.imwrite('cuttest.tif',cropped)
cropped1 = im_data1[ 32:80,131:179]
#cropped1 = im_data1[XYdr[i][0]:XYul[i][0], XYdr[i][1]:XYul[i][1]]
cropped2 = im_data2[XYul[0][0]:XYdr[0][0], XYul[0][1]:XYdr[0][1]]
cropped3 = im_data3[XYul[0][0]:XYdr[0][0], XYul[0][1]:XYdr[0][1]]
#cropped4 = im_data4[XYul[0][0]:XYdr[0][0], XYul[0][1]:XYdr[0][1]]

print(XYul,XYdr)
plt.imshow(cropped1)
plt.xticks([]), plt.yticks([])  # 不显示坐标轴
plt.show()

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1,np2,np3):
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
   # out_band = out_ds.GetRasterBand(4)
   # out_band.WriteArray(np4)
    return

label = open('green3.txt', 'w')
# print(tensor1)
for i in range(0,len(GCPdata)):
    nnpp1 = im_data1[XYul[i][1]:XYdr[i][1],XYul[i][0]:XYdr[i][0]]
    nnpp2 = im_data2[XYul[i][1]:XYdr[i][1],XYul[i][0]:XYdr[i][0]]
    nnpp3 = im_data3[XYul[i][1]:XYdr[i][1],XYul[i][0]:XYdr[i][0]]
    #nnpp4 = im_data4[XYul[i][1]:XYdr[i][1],XYul[i][0]:XYdr[i][0]]
    write_img('green_e_3_'+str(2*RECT_SIZE)+'\\'+'test_'+str(i)+'_0314_3.tif', 2*RECT_SIZE, 2*RECT_SIZE, 3, band1.DataType,nnpp1,nnpp2,nnpp3)
    label.write('D:\why2022\seaice\data_process\\green_e_3_'+str(2*RECT_SIZE)+'\\'+'test_'+str(i)+'_0314_3.tif'+' '+'3\n')

# plt.imshow(cropped1)
# plt.xticks([]), plt.yticks([])  # 不显示坐标轴
# plt.show()
label.close()