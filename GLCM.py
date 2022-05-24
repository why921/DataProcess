from skimage.feature import greycomatrix, greycoprops
import numpy as np
#"D:\why2022\DL\SAR_specific_models-master\SAR_specific_models-master\data\slc_data\agriculture\agriculture_4312_HH_11653_2005.npy"
import matplotlib.pyplot as plt



image = np.load('D:\why2022\DL\SAR_specific_models-master\SAR_specific_models-master\data\slc_data\\agriculture\\agriculture_4312_HH_11653_2005.npy')
#plt.imshow(image)
I=np.zeros((20,20))
a=np.ones((20,20))

I[a==1]=3
print(I)

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48_4bands\ALPSRP267211510_spe48_0_4b_0.tif"
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

path1 = (lines[1]).split(" ")[0]
path2 = (lines[1]).split(" ")[0]
path3 = (lines[1]).split(" ")[0]
path4 = (lines[1]).split(" ")[0]
print(path1)
ds1 = gdal.Open(str(path1))
band1 = ds1.GetRasterBand(1)
spe1 = band1.ReadAsArray()
img = np.array([spe1, spe1,spe1])
DDSS=gdal.Open(r'E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli48\test_500and100.tif')
band11 = DDSS.GetRasterBand(1)
#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\slc48\SLC_1000_200.tif"
#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli48\test_500and100.tif"
print(band11.DataType)





