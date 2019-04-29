import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class loadDataset(object):
	def __init__(self,tp):
		pass
	@staticmethod
	def getData(inp_path,img_size,dataAug,testing,num_channel=1):
		if(inp_path==None):
			PATH = os.getcwd()
			data_path = os.path.join(PATH,'data')
			data_dir_list = os.listdir(data_path)
		else:
			data_path=inp_path
			data_dir_list=os.listdir(inp_path)
		num_classes=len(data_dir_list)
		print(num_classes)
		num_samples = 0
		img_data_list=[]
		label_list=[]
		curr_label=0
		for class_no in data_dir_list:
			class_path=os.path.join(data_path,class_no)
			imglist=os.listdir(class_path)
			for img in imglist:
				try:
					input_img=cv2.imread(os.path.join(class_path,img))
					input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
					input_img_resize=cv2.resize(input_img,img_size)
					img_data_list.append(input_img_resize)
					label_list.append(curr_label)
					num_samples+=1
				except:
					print("couldn't open image file "+os.path.join(class_path,img))
			curr_label+=1
		
		img_data = np.array(img_data_list)
		img_data = img_data.astype('float32')
		img_data /= 255
		Y = np_utils.to_categorical(label_list, num_classes)
		if dataAug:
			datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
		else:
			datagen=None
		if num_channel==1:
			if K.image_dim_ordering()=='th':
				img_data= np.expand_dims(img_data, axis=1) 
				print (img_data.shape)
			else:
				img_data= np.expand_dims(img_data, axis=4) 
				print (img_data.shape)
				
		else:
			if K.image_dim_ordering()=='th':
				img_data=np.rollaxis(img_data,3,1)
				print (img_data.shape)

		images,labels = shuffle(img_data,Y)
		if testing:
			X_test,y_test=images,labels
			X_train,y_train=None,None
		else:
			X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

		return  datagen,X_train, X_test, y_train, y_test,num_classes










