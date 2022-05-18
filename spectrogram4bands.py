import numpy as np
from osgeo import gdal
from scipy import fftpack
import cv2

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48.txt"

spectrogramtxt = open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_48.txt','r')

lines=spectrogramtxt.readlines()
print(len(lines))
print(lines[467])

for i in range(0,len(spectrogramtxt)):
    path=1