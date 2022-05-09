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

# E:\ALOSPALSAR\Greenland201101\510
# "E:\ALOSPALSAR\Greenland201101\510\Data0509.txt"
GCPdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2, 7))
XYdata = np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(1, 2))
label=np.loadtxt('E:\ALOSPALSAR\Greenland201101\\510\Data0509.txt', dtype=int, skiprows=6, usecols=(7))
XYsize = np.ones_like(GCPdata[:, 0:2])
print(label[99])
print(XYsize)
print(GCPdata)
print(label)
