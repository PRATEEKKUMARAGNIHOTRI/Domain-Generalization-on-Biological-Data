# -*- coding: utf-8 -*-
"""download_and_process_mnist.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OAImw-aZ_pOBVCF2jgqM7M1favHgGcIR
"""

import pandas as pd
import numpy as np
import scipy.misc
import time
import PIL
import os
import cPickle
from PIL import Image

"""
   Requirements:
          Folder - images
          Files  - train_details.csv, train.csv, test.csv
          """

#########list cam3_images names of images from cam3 and list label_cam3 has corresponding labels#############
#########Same for cam2###############

cam3_images=[]
label_cam3 = []
cam2_images=[]
label_cam2 = []

#########################################################################################################

classes = {'nrbc': 0, 'notawbc': 1 , 'giant platelet': 2, 'platelet clump': 3, 'basophil':4 ,
			'neutrophil': 5, 'eosinophil': 6, 'lymphocyte': 7, 'monocyte': 8, 'ig': 9, 'atypical-blast': 10}

#########################################################################################################


data = pd.read_csv("./train_details.csv")
headers = list(data)

a = pd.read_csv("./train.csv").to_numpy()
c = pd.read_csv("./test.csv").to_numpy()

print(a.shape)
print(c.shape)

images = data[headers[0]].values.tolist()
cam = data[headers[1]].values.tolist()
labels = data[headers[2]].values.tolist()

for i in range(len(cam)):
  if "cam2"==cam[i].lower():
    cam2_images.append(images[i])

  if "cam3"==cam[i].lower():
    cam3_images.append(images[i])

print('No. of cam3 images: ', len(cam3_images))
print('No. of cam2 images: ', len(cam2_images))

np.random.shuffle(cam3_images)
np.random.shuffle(cam2_images)


a = np.concatenate((a,c))
np.random.shuffle(a)
b = {}

for i in range(a.shape[0]):
  b[a[i][0]]=a[i][1]

for i in cam3_images:
      label_cam3.append(classes[b[i]])


for i in cam2_images:
      label_cam2.append(classes[b[i]])

######################################################################

pt=0.1

test = {'X': np.empty((int(pt*len(cam3_images)),128,128,3)) , 'y': np.array(label_cam3[:int(pt*len(cam3_images))])}
train = {'X': np.empty((len(cam3_images)-int(pt*len(cam3_images)),128,128,3)) , 'y': np.array(label_cam3[int(pt*len(cam3_images)):])}


######################################################################

for i in range(int(pt*len(cam3_images))):
      test[i] = np.array(PIL.Image.open('./images/'+cam3_images[i]))

for i in range(int(pt*len(cam3_images)),len(cam3_images)):
      train[i] = np.array(PIL.Image.open('./images/'+cam3_images[i]))
      
######################################################################

np.save('./data/mnist/test.npy',test['X'])
np.save('./data/mnist/test_label.npy',test['y'])
np.save('./data/mnist/train.npy',train['X'])
np.save('./data/mnist/train_label.npy',train['y'])


######################################################################

######################################################################


train = {'X': np.empty((len(cam2_images),128,128,3)) , 'y': np.array(label_cam2)}

######################################################################

for i in range(len(cam2_images)):
      test[i] = np.array(PIL.Image.open('./images/'+cam2_images[i]))
  


######################################################################

np.save('./data/svhn/train.npy',train['X'])
np.save('./data/svhn/train_label.npy',train['y'])