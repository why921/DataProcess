import numpy as np
import os
import torch
from osgeo import gdal
from img_statistics import ALPSRP205991510


sar='ALPSRP201761520'
RECT_SIZE=24
#"E:\ALOSPALSAR\Beaufort\91510\ALOS-P1_1__A-ORBIT__ALPSRP205991510_Cal_ML_Spk_Decomp.tif"
#"E:\ALOSPALSAR\Beaufort\91510\ALOS-P1_1__A-ORBIT__ALPSRP205991510_Cal_ML.tif"

#"E:\ALOSPALSAR\Beaufort\61520\ALOS-P1_1__A-ORBIT__ALPSRP201761520_Cal_ML_Spk_Decomp.tif"
ds = gdal.Open(r'E:\ALOSPALSAR\Beaufort\61520\ALOS-P1_1__A-ORBIT__ALPSRP201761520_Cal_ML_Spk_Decomp.tif')
slc = gdal.Open(r'E:\ALOSPALSAR\Beaufort\61520\ALOS-P1_1__A-ORBIT__ALPSRP201761520_Cal_ML.tif')

label = open('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\pauli'+str(2*RECT_SIZE)+'.txt', 'w',encoding='utf-8')
labeltxt=open('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\slc'+str(2*RECT_SIZE)+'.txt', 'w',encoding='utf-8')

rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

im_data1 = band1.ReadAsArray()  # 获取数据
im_data2 = band2.ReadAsArray()  # 获取数据
im_data3 = band3.ReadAsArray()  # 获取数据

im_data1 = (im_data1 - (im_data1.min())) / ((im_data1.max()) - (im_data1.min()))
im_data2 = (im_data2 - (im_data2.min())) / ((im_data2.max()) - (im_data2.min()))
im_data3 = (im_data3 - (im_data3.min())) / ((im_data3.max()) - (im_data3.min()))


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
    return



srows = slc.RasterYSize
scols = slc.RasterXSize
sbands = slc.RasterCount

sband1 = slc.GetRasterBand(1)
sband2 = slc.GetRasterBand(2)
sband3 = slc.GetRasterBand(3)
sband4 = slc.GetRasterBand(4)
sband5 = slc.GetRasterBand(5)
sband6 = slc.GetRasterBand(6)
sband7 = slc.GetRasterBand(7)
sband8 = slc.GetRasterBand(8)

slc_data1 = sband1.ReadAsArray()  # 获取数据
slc_data2 = sband2.ReadAsArray()  # 获取数据
slc_data3 = sband3.ReadAsArray()  # 获取数据
slc_data4 = sband4.ReadAsArray()  # 获取数据
slc_data5 = sband5.ReadAsArray()  # 获取数据
slc_data6 = sband6.ReadAsArray()  # 获取数据
slc_data7 = sband7.ReadAsArray()  # 获取数据
slc_data8 = sband8.ReadAsArray()  # 获取数据

imgHH=slc_data1+1j*slc_data2
imgHV=slc_data3+1j*slc_data4
imgVH=slc_data5+1j*slc_data6
imgVV=slc_data7+1j*slc_data8

def write_slc(filename, XSIZE, YSIZE, Bands, DataType, np1,np2,np3,np4):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_ds.SetProjection(slc.GetProjection())
    out_ds.SetGeoTransform(slc.GetGeoTransform())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    out_band = out_ds.GetRasterBand(2)
    out_band.WriteArray(np2)
    out_band = out_ds.GetRasterBand(3)
    out_band.WriteArray(np3)
    out_band = out_ds.GetRasterBand(4)
    out_band.WriteArray(np4)
    return
#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli48.txt"

#E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli48
for i in range(150, 450):
    for j in range(0, 300):
        cut1 = im_data1[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]
        cut2 = im_data2[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]
        cut3 = im_data3[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]

        nnpp1 = imgHH[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]
        nnpp2 = imgHV[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]
        nnpp3 = imgVH[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]
        nnpp4 = imgVV[2*i:2*i + 2*RECT_SIZE, 2*j:2*j + 2*RECT_SIZE]


        write_img('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\pauli'+str(2*RECT_SIZE)+'\\'+'test_'+str(i)+'and'+str(j)+'.tif', 2*RECT_SIZE, 2*RECT_SIZE, 3, band1.DataType,cut1,cut2,cut3)
        label.write('E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\pauli'+str(2*RECT_SIZE)+'\\'+'test_'+str(i)+'and'+str(j)+'.tif'+' '+'0\n')

        write_slc(
            'E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\slc'+str(2*RECT_SIZE) + '\\' + 'SLC_' + str(
                2*i) + '_' + str(2*j) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,4, sband1.DataType, nnpp1,nnpp2,nnpp3,nnpp4)

        labeltxt.write(
            'E:\ALOSPALSAR\ValidationData'+'\\'+sar+'\\slc'+str(2*RECT_SIZE) + '\\' + 'SLC_' + str(
                2*i) + '_' + str(2*j) + '.tif' + ' ' + str(0) + '\n')




label.close()
labeltxt.close()

