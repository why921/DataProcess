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


win_size=12
num=0
#bartlett hanning hamming kaiser
hamming_win = np.kaiser(win_size,4)
hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
#E:\DataSetProcess\gtxt.txt
gtxt=open('E:\DataSetProcess\gtxt.txt', 'w')
def show(img):
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
#"E:\ALOSPALSAR\imgshow\Beaufort91510SLC\ALPSRP267211510_SLC_HH_0_0509_2.tif"
#"E:\ALOSPALSAR\imgshow\Beaufort51550SLC\ALPSRP267211510_SLC_HH_0_0509_2.tif"
ds = gdal.Open('E:\ALOSPALSAR\imgshow\Beaufort51550SLC\ALPSRP267211510_SLC_HH_0_0509_2.tif')
band1 = ds.GetRasterBand(1)
slc_data1 = band1.ReadAsArray()

slc_fft = fftpack.fftshift(fftpack.fft2(slc_data1))
spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)
ff = np.zeros((win_size, win_size))
for i in range(0, win_size):
    for j in range(0, win_size):
        spectrogram[:, :, i, j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i + win_size, j:j + win_size],
                                                        shape=[win_size, win_size])
spectrogram = np.log(1 + np.abs(spectrogram))
x, y, fr, fa = spectrogram.shape
spectrogram = spectrogram.reshape([x *y, fr, fa])
IMG=[]
for K in range(0, (x * y)):
    ff += spectrogram[K, :, :]
    #E:\ALOSPALSAR\imgshow\temp
    cv2.imwrite('E:\ALOSPALSAR\imgshow\\temp' + '\\' + 'img_' + str(num) + '.tif', ff)
    gtxt.write('E:\ALOSPALSAR\imgshow\\temp' + '\\' + 'img_' + str(num) + '.tif'+' '+str(0)+'\n')
    num=num+1
gtxt.close()

#torchvision.utils.save_image(img_tensor, 'out.jpg')
class gridDataset(Dataset):
    def __init__(self, labeltxt, transform, target_transform=None):
        fh = open(labeltxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        ds = gdal.Open(fn)
        band1 = ds.GetRasterBand(1)
        im_data1 = band1.ReadAsArray()
        im_data1 = (im_data1 - (im_data1.min())) / ((im_data1.max()) - (im_data1.min()))
        img = np.array([im_data1])

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

data_transforms = transforms.Compose([
   DataTrans.Numpy2Tensor(),
])


ddss = gridDataset(labeltxt='E:\DataSetProcess\gtxt.txt',transform=data_transforms)
ddss.__init__(labeltxt='E:\DataSetProcess\gtxt.txt',transform=data_transforms)


dataloader = torch.utils.data.DataLoader(ddss,
                                    batch_size=144, # 批量大小
                                    shuffle=False # 多进程
                                     )
xxx, label = iter(dataloader).next()

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