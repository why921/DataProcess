import numpy as np
from osgeo import gdal
from createTrainData import label_path
from createTrainData import sar
from createTrainData import date

path='E:\ALOSPALSAR\TrainData'
img_path='E:\ALOSPALSAR\Beaufort\\51550\ALOS-P1_1__A-ORBIT__ALPSRP258351550_Cal_ML.tif'
txt_path='E:\ALOSPALSAR\TrainData\ALPSRP258351550\ALPSRP258351550_SLC_24.txt'


GCPdata = np.loadtxt(label_path, dtype=int, skiprows=6, usecols=(1, 2, 7))
XYdata = np.loadtxt(label_path, dtype=int, skiprows=6, usecols=(1, 2))
label = np.loadtxt(label_path, dtype=int, skiprows=6, usecols=(7))
XYsize = np.ones_like(GCPdata[:, 0:2])

RECT_SIZE = 12

XYul = XYdata - RECT_SIZE * XYsize
XYdr = XYdata + RECT_SIZE * XYsize

ds = gdal.Open(img_path)

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

im_data1 = band1.ReadAsArray()
im_data2 = band2.ReadAsArray()
im_data3 = band3.ReadAsArray()
im_data4 = band4.ReadAsArray()
im_data5 = band5.ReadAsArray()
im_data6 = band6.ReadAsArray()
im_data7 = band7.ReadAsArray()
im_data8 = band8.ReadAsArray()

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

labeltxt = open(txt_path, 'w')

for i in range(0, len(GCPdata)):
    nnpp1 = imgHH[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp2 = imgHV[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp3 = imgVH[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp4 = imgVV[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    labelid=label[i]
    write_img(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'HH_' + str(i) + date + str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp1)
    labeltxt.write(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'HH_' + str(i) + date + str(labelid)+'.tif' + ' ' + str(labelid)+'\n')
    write_img(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'HV_' + str(i) + date + str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp2)
    labeltxt.write(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'HV_' + str(i) + date + str(labelid)+'.tif' + ' ' + str(labelid)+'\n')
    write_img(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'VH_' + str(i) + date + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp3)
    labeltxt.write(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'VH_' + str(i) + date + str(labelid) + '.tif' + ' ' + str(labelid) + '\n')
    write_img(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'VV_' + str(i) + date + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
              1, band1.DataType, nnpp4)
    labeltxt.write(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_SLC_'+'VV_' + str(i) + date + str(labelid) + '.tif' + ' ' + str(labelid) + '\n')

labeltxt.close()