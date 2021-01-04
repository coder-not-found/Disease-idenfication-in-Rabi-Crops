

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import os
os.environ['KAGGLE_USERNAME'] = "rutulgandhi05" # username from the json file
os.environ['KAGGLE_KEY'] = "f12da783ad6436498ae4903810b7d407" # key from the json file
!kaggle datasets download -d emmarex/plantdisease # api copied from kaggle

from zipfile import ZipFile

file_name = "/content/plantdisease.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

pip install split-folders tqdm

import split_folders


split_folders.ratio('PlantVillage', output="output", seed=1337, ratio=(.8, .1, .1))

IMAGE_SIZE = [224, 224]

train_path = '/content/output/train'
valid_path = '/content/output/val'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#not training existing waits
for layer in vgg.layers:
  layer.trainable = False

folders = glob('/content/output/train/*')

out = Flatten()(vgg.output)
out = Dropout(0.55)(out)
out = Dense(512, activation='relu')(out)
predictions = Dense(len(folders), activation='softmax')(out)
model = Model(inputs=vgg.input, outputs=predictions)

model.summary()

EPOCHS = 30
INIT_LR = 1e-4

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
  loss='categorical_crossentropy',
  optimizer=opt,
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/output/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/output/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=EPOCHS,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import tensorflow as tf

from keras.models import load_model

model.save('plantdisease_model.h5')

test_generator = test_datagen.flow_from_directory('/content/output/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

scores = model.evaluate_generator(test_generator) #1514 testing images
print("Test Accuracy = ", scores[1])

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

predictions = model.predict_generator(test_generator)

from sklearn import metrics

y_pred = np.argmax(predictions, axis=-1)
y_test = test_generator.classes

print("precision Score", metrics.precision_score(y_test, y_pred , average="macro"))
print("Recall", metrics.recall_score(y_test, y_pred , average="macro"))
print("f1 Score", metrics.f1_score(y_test, y_pred , average="macro"))



auc= metrics.roc_auc_score(y_test, y_pred,multi_class='ovo')

print(metrics.classification_report(val_trues, val_preds))

