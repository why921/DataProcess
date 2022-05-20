import numpy as np
from osgeo import gdal
from scipy import fftpack
import cv2

# "E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48.txt"
ds = gdal.Open(r'E:\ALOSPALSAR\Greenland201101\510\ALOS-P1_1__A-ORBIT__ALPSRP267211510_Cal_ML.tif')

spectrogramtxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48.txt', 'r')
lines = spectrogramtxt.readlines()
print(len(lines))
print(lines[467])
size=24



#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48_4bands.txt"
spetxt=open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48_4bands.txt', 'w')
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



for i in range(0, int(len(lines) / 4)):
    pathHH = (lines[i]).split(" ")[0]
    pathHV = (lines[i+1]).split(" ")[0]
    pathVH = (lines[i+2]).split(" ")[0]
    pathVV = (lines[i+3]).split(" ")[0]
    label = (lines[i*4]).split(" ")[1]
    labelid=int(label)

    dsHH = gdal.Open(str(pathHH))
    bandHH = dsHH.GetRasterBand(1)
    speHH = bandHH.ReadAsArray()

    dsHV = gdal.Open(str(pathHV))
    bandHV = dsHV.GetRasterBand(1)
    speHV = bandHV.ReadAsArray()

    dsVH = gdal.Open(str(pathVH))
    bandVH = dsVH.GetRasterBand(1)
    speVH = bandVH.ReadAsArray()

    dsVV = gdal.Open(str(pathVV))
    bandVV = dsVV.GetRasterBand(1)
    speVV = bandVV.ReadAsArray()

    write_img('E:\\ALOSPALSAR\\TrainData\\ALPSRP267211510\\ALPSRP267211510_spe_48_4bands\\'+'ALPSRP267211510_spe48_'+str(i)+'_4b_'+str(labelid)+'.tif',
              size,size,4,bandHH.DataType,speHH,speHV,speVH,speVV)
    spetxt.write('E:\\ALOSPALSAR\\TrainData\\ALPSRP267211510\\ALPSRP267211510_spe_48_4bands\\'+'ALPSRP267211510_spe48_'+str(i)+'_4b_'+str(labelid)+'.tif'
                 + ' ' + str(label))



spetxt.close()
