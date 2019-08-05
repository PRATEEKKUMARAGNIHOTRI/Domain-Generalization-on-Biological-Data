import pandas as pd
import numpy as np
import scipy.misc
import time
import PIL
import os
import cPickle
from PIL import Image
import csv

"""
   Requirements:
          Folder - images
          Files  - train_details.csv, train.csv, test.csv
          """

classes = {'nrbc': 0, 'notawbc': 1 , 'giant platelet': 2, 'platelet clump': 3, 'basophil':4 ,
			'neutrophil': 5, 'eosinophil': 6, 'lymphocyte': 7, 'monocyte': 8, 'ig': 9, 'atypical-blast': 10}

###############################################################

my_dict={}

with open('./train.csv') as f_input:
    for row in csv.reader(f_input):
        my_dict[row[0]] = row[1]
        
data = pd.read_csv("./train_details.csv").to_numpy()

labels_cam3 = []
images_cam3 = []
labels_cam2 = []
images_cam2 = []

for row in data:
  if(row[1]=='cam3'):
    img = row[0]
    images_cam3.append(np.array(PIL.Image.open('./images/'+row[0])))
    labels_cam3.append(classes[my_dict[img]])

np.save("./data/mnist/train.npy",np.array(images_cam3))
np.save("./data/mnist/train_label.npy",np.array(labels_cam3))

x = list(np.load("./data/mnist/train_label.npy"))

y = [0]*11
for i in range(len(x)):
  y[x[i]] = y[x[i]]+1

print("Cam3_train",y)

for row in data:
  if(row[1]=='cam2'):
    img = row[0]
    images_cam2.append(np.array(PIL.Image.open('./images/'+row[0])))
    labels_cam2.append(classes[my_dict[img]])

np.save("./data/svhn/train.npy",np.array(images_cam2))
np.save("./data/svhn/train_label.npy",np.array(labels_cam2))

x = np.load("./data/svhn/train_label.npy")
y = [0]*11
for i in range(len(x)):
  y[x[i]] = y[x[i]]+1
print("Cam2_train",y)


my_dict={}

with open('./test.csv') as f_input:
    for row in csv.reader(f_input):
        my_dict[row[0]] = row[1]
        
data = pd.read_csv("./test_details.csv").to_numpy()

labels_cam3 = []
images_cam3 = []
labels_cam2 = []
images_cam2 = []

for row in data:
  if(row[1]=='cam3'):
    img = row[0]
    images_cam3.append(np.array(PIL.Image.open('./images/'+row[0])))
    labels_cam3.append(classes[my_dict[img]])

np.save("./data/mnist/test.npy",np.array(images_cam3))
np.save("./data/mnist/test_label.npy",np.array(labels_cam3))

x = np.load("./data/mnist/test_label.npy")
y = [0]*11
for i in range(len(x)):
  y[x[i]] = y[x[i]]+1
print("Cam3_test",y)


for row in data:
  if(row[1]=='cam2'):
    img = row[0]
    images_cam2.append(np.array(PIL.Image.open('./images/'+row[0])))
    labels_cam2.append(classes[my_dict[img]])

np.save("./data/svhn/test.npy",np.array(images_cam2))
np.save("./data/svhn/test_label.npy",np.array(labels_cam2))

x = np.load("./data/svhn/test_label.npy")
y = [0]*11
for i in range(len(x)):
  y[x[i]] = y[x[i]]+1
  
print("cam2 Test",y)
