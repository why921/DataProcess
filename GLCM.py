from skimage.feature import greycomatrix, greycoprops
import numpy as np
#"D:\why2022\DL\SAR_specific_models-master\SAR_specific_models-master\data\slc_data\agriculture\agriculture_4312_HH_11653_2005.npy"
import matplotlib.pyplot as plt



image = np.load('D:\why2022\DL\SAR_specific_models-master\SAR_specific_models-master\data\slc_data\\agriculture\\agriculture_4312_HH_11653_2005.npy')
#plt.imshow(image)

print(image.shape)