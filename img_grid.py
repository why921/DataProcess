import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from osgeo import gdal
from scipy import fftpack
import cv2
import torch
import DataTrans
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

win_size=12
num=0
hamming_win = np.hamming(win_size)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
#E:\DataSetProcess\gtxt.txt
gtxt=open('E:\DataSetProcess\gtxt.txt', 'w')
#ALPSRP267211510_spectrogram_48
def show(img):
    """
    用来显示图片的
    """
    plt.figure(figsize=(12, 12))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def write_img(filename, XSIZE, YSIZE, Bands, DataType, np1):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(filename,
                                 XSIZE, YSIZE, Bands, DataType)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np1)
    return
#write_img('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_48' + '\\' + 'ALPSRP267211510_SLC_VV_' + str(i) + '_0509_' + str(labelid) + '.tif', 2 * RECT_SIZE, 2 * RECT_SIZE,
        #      3, band1.DataType, nnpp4)

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_24_4bands\ALPSRP267211510_spe24_0_4b_0.tif"
ds = gdal.Open('E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_SLC_24\ALPSRP267211510_SLC_HH_0_0509_0.tif')
band1 = ds.GetRasterBand(1)
slc_data1 = band1.ReadAsArray()  # 获取数据
slc_fft = fftpack.fftshift(fftpack.fft2(slc_data1))
spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
ff = np.zeros((win_size, win_size))
for i in range(0, win_size):
    for j in range(0, win_size):
        spectrogram[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i + win_size, j:j + win_size],
                                                        shape=[win_size, win_size])
spectrogram = np.log(1 + np.abs(spectrogram))
x, y, fr, fa = spectrogram.shape
spectrogram = spectrogram.reshape([x * y, fr, fa])
IMG=[]
for K in range(0, (x * y)):
    ff += spectrogram[K, :, :]
    #cv2.imwrite('E:\DataSetProcess\grid' + '\\' + 'img_' + str(num) + '.jpg', ff)
    gtxt.write('E:\DataSetProcess\grid' + '\\' + 'img_' + str(num) + '.tif'+' '+str(0)+'\n')
    num=num+1
gtxt.close()

#torchvision.utils.save_image(img_tensor, 'out.jpg')
class pauliDataset(Dataset):
    def __init__(self, labeltxt, transform, target_transform=None):
        fh = open(labeltxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存
            # words[0]图片，words[1]lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn图片path #fn和label分别获得imgs[index]即每行中word[0]和word[1]的信息
        ds = gdal.Open(fn)
        band1 = ds.GetRasterBand(1)
        im_data1 = band1.ReadAsArray()
        im_data1 = (im_data1 - (im_data1.min())) / ((im_data1.max()) - (im_data1.min()))
        img = np.array([im_data1])

        if self.transform is not None:
            img = self.transform(img)
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

data_transforms = transforms.Compose([
   DataTrans.Numpy2Tensor(),
])

#"E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_48.txt"
ddss = pauliDataset(labeltxt='E:\DataSetProcess\gtxt.txt',transform=data_transforms)
ddss.__init__(labeltxt='E:\DataSetProcess\gtxt.txt',transform=data_transforms)
print(ddss.__len__())
img, gt = ddss.__getitem__(2) # get the 34th sample
print(type(img))
print(img)
print(gt)
dataloader = torch.utils.data.DataLoader(ddss,
                                    batch_size=144, # 批量大小

                                    shuffle=False # 多进程
                                     )
xxx, label = iter(dataloader).next()
print(xxx)
print('xxx:', xxx.shape, 'label:', label.shape)
show(make_grid(xxx, nrow=12, padding=0))
torchvision.utils.save_image(xxx, 'test.png', nrow=12, padding=0)
'''
ff = ss[1,:, :, :]
plt.imshow(ff
plt.xticks([]), plt.yticks([])  # 不显示坐标轴
plt.show()
'''



'''
img_tensor = transforms.ToTensor()(spectrogram)
img_tensor = torchvision.utils.make_grid(img_tensor)
torchvision.utils.save_image(img_tensor, 'out.jpg')
'''

'''
img = plt.imread('wave.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.repeat(10, 1, 1, 1)
img_tensor = torchvision.utils.make_grid(img_tensor)
torchvision.utils.save_image(img_tensor, 'out.jpg')
'''