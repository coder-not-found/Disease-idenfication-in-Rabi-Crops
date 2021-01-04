# disease-identification-in-rabi-crops
this repository contains the codes of the different models which we used to tackle the problem(DISEASE IDENTIFICATION IN RABI CROPS) which was given to us as a part of our internship under leading India AI and Bennett University. 
the models which we used are : 

•	CNN
•	Vgg 16
•	ResNET
•	DenseNET
•	AlexNET
•	Inception

Requirements : 
•	import numpy as np
•	import pickle
•	import cv2
•	from os import listdir
•	from sklearn.preprocessing import LabelBinarizer
•	from keras.models import Sequential
•	from keras.layers.normalization import BatchNormalization
•	from keras.layers.convolutional import Conv2D
•	from keras.layers.convolutional import MaxPooling2D
•	from keras.layers.core import Activation, Flatten, Dropout, Dense
•	from keras import backend as K
•	from keras.preprocessing.image import ImageDataGenerator
•	from keras.optimizers import Adam
•	from keras.preprocessing import image
•	from keras.preprocessing.image import img_to_array
•	from sklearn.preprocessing import MultiLabelBinarizer
•	from sklearn.model_selection import train_test_split
•	import matplotlib.pyplot as plt
•	from keras.models import Model
•	from keras.optimizers import Adam
•	from keras.layers import GlobalAveragePooling2D
•	from keras.layers import Dense
•	from keras.applications.inception_v3 import InceptionV3
•	from keras.utils.np_utils import to_categorical
•	from keras.applications.vgg16 import VGG16
•	from keras.applications.vgg16 import preprocess_input
•	import seaborn as sns
•	from keras.applications import DenseNet121


TO RUN THE FILES 
•	TO ACCESS CNN FILE: RUN disease identification using CNN.py
•	TO ACCESS VGG16 FILE: RUN disease identification using VGG16.py
•	TO ACCESS INCEPTION FILE: RUN disease identification using Inception.py
•	TO ACCESS DENSENET FILE: RUN disease identification using DenseNET.py
•	TO ACCESS ALEXNET FILE: RUN disease identification using alexnet.py
•	TO ACCESS RESNET FILE: RUN disease identification using resnet.py
