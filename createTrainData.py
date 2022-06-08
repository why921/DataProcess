import numpy as np
from osgeo import gdal



sar='ALPSRP258351550'
date='_0608_'
path='E:\ALOSPALSAR\TrainData'

img_path='E:\ALOSPALSAR\Beaufort\\51550\\ALOS-P1_1__A-ORBIT__ALPSRP258351550_Cal_ML_Spk_Decomp.tif'
label_path='E:\ALOSPALSAR\Beaufort\\51550\Data51550.txt'



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

im_data1 = band1.ReadAsArray()
im_data2 = band2.ReadAsArray()
im_data3 = band3.ReadAsArray()

im_data1 = (im_data1 - (im_data1.min())) / ((im_data1.max()) - (im_data1.min()))
im_data2 = (im_data2 - (im_data2.min())) / ((im_data2.max()) - (im_data2.min()))
im_data3 = (im_data3 - (im_data3.min())) / ((im_data3.max()) - (im_data3.min()))



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

labeltxt = open(path+'\\'+sar+'\\'+sar+'_24.txt', 'w')

for i in range(0, len(GCPdata)):
    nnpp1 = im_data1[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp2 = im_data2[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    nnpp3 = im_data3[XYul[i][1]:XYdr[i][1], XYul[i][0]:XYdr[i][0]]
    labelid=label[i]
    write_img(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_' + str(i) + str(date)+str(labelid)+'.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
             3, band1.DataType, nnpp1, nnpp2, nnpp3)
    labeltxt.write(path+'\\'+sar+'\\'+sar+'_24\\'+sar+'_' + str(i) + str(date)+str(labelid)+'.tif' + ' ' + str(labelid)+'\n')

labeltxt.close()

#ALPSRP267211510 r -39.0113639831543 -13.956258773803711  g -42.140995025634766 -15.663641929626465 b -36.93307876586914 -4.1313347816467285