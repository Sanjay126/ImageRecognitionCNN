
# coding: utf-8

# In[6]:


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[8]:


img_rows = 256
img_cols = 256
channels = 1

pathin='F:/NNFLproject/Leaves'
pathto='F:/NNFLproject/Input'

listing = os.listdir(pathin)
num_samples = len(listing)
print(num_samples)


# In[10]:


for file in listing:
    img1 = Image.open(pathin+'\\'+file)
    img2 = img1.resize((img_rows,img_cols))
    gscale = img2.convert('L')
    gscale.save(pathto+'\\'+file,"JPEG")    


# In[16]:


imlist = os.listdir(pathto)
im1 = np.array(Image.open(pathto+'\\'+imlist[0])) #open one image to get size
m,n = im1.shape[0:2] #get size of images
img_no = len(imlist)
imageMat = np.array([np.array(Image.open(pathto+'\\'+im2)).flatten()
                 for im2 in imlist],'f')


# In[47]:


label = np.ones((num_samples,),dtype=int)
label[0:59] = 0
label[59:122] = 1
label[122:194] = 2
label[194:267] = 3
label[267:323] = 4
label[323:385] = 5
label[385:437] = 6
label[437:496] = 7
label[496:551] = 8
label[551:616] = 9
label[616:666] = 10
label[666:729] = 11
label[729:781] = 12
label[781:846] = 13
label[846:906] = 14
label[906:962] = 15
label[962:1039] = 16
label[1039:1101] = 17
label[1101:1162] = 18
label[1162:1228] = 19
label[1228:1288] = 20
label[1288:1343] = 21
label[1343:1398] = 22
label[1398:1463] = 23
label[1463:1517] = 24
label[1517:1569] = 25
label[1569:1622] = 26
label[1622:1677] = 27
label[1677:1734] = 28
label[1734:1798] = 29
label[1798:1851] = 30
label[1851:1907] = 31


# In[51]:


data,label = shuffle(imageMat,label,random_state=2)
train_data = [data,label]


# In[57]:


temp = imageMat[0].reshape(img_rows,img_cols)
plt.imshow(temp)
plt.show()
print(train_data[0].shape)


# In[ ]:


batch_size = 80
num_classes = 32
epochs = 10
num_filters = 32
num_conv = 5
stride = 2
num_pool = 2

