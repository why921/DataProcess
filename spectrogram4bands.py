from osgeo import gdal
from creatSLC import img_path
from createTrainData import sar


path='E:\ALOSPALSAR\TrainData'
spe_path='E:\ALOSPALSAR\TrainData\ALPSRP258351550\ALPSRP258351550_spe_24.txt'
spe4b_path='E:\ALOSPALSAR\TrainData\ALPSRP258351550\ALPSRP258351550_spe_24_4bands.txt'



ds = gdal.Open(img_path)

spectrogramtxt = open(path+'\\'+sar+'\\'+sar+'_spe_24.txt', 'r')
spe4btxt=open(path+'\\'+sar+'\\'+sar+'_spe_24_4bands.txt', 'w')

lines = spectrogramtxt.readlines()

size=12



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

    write_img(path+'\\'+sar+'\\'+sar+'_spe_24_4bands\\'+sar+'_spe24_'+str(i)+'_4b_'+str(labelid)+'.tif',size,size,4,bandHH.DataType,speHH,speHV,speVH,speVV)
    spe4btxt.write(path+'\\'+sar+'\\'+sar+'_spe_24_4bands\\'+sar+'_spe24_'+str(i)+'_4b_'+str(labelid)+'.tif'+ ' ' + str(label))

spe4btxt.close()
