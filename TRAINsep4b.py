from osgeo import gdal
from createTrainData import sar
from createTrainData import SLC_img_path
from createTrainData import RECT_SIZE

path='E:\ALOSPALSAR\TrainData'


ds = gdal.Open(SLC_img_path)

spectrogramtxt = open("E:\\ALOSPALSAR\\TrainData\\spe18.txt", 'r')
spe4btxt=open("E:\\ALOSPALSAR\\TrainData\\spe4bands18.txt", 'w')

lines = spectrogramtxt.readlines()

size=18



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

    write_img('E:\\ALOSPALSAR\\TrainData\\spe4bands18'+'\\'+'spe'+str(2*RECT_SIZE)+'_'+str(i)+'_4b_'+str(labelid)+'.tif',size,size,4,bandHH.DataType,speHH,speHV,speVH,speVV)
    spe4btxt.write('E:\\ALOSPALSAR\\TrainData\\spe4bands18'+'\\'+'spe'+str(2*RECT_SIZE)+'_'+str(i)+'_4b_'+str(labelid)+'.tif'+ ' ' + str(label))

spe4btxt.close()
