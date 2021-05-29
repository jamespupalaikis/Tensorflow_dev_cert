import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


'''import json
with open('data.json', 'r') as f:
    train = json.load(f)
'''
data_dir = 'data/asl_alphabet_train'
batch_size = 500
resize = (50,50)

datagen = training_datagen = ImageDataGenerator()
'''rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.4,
      horizontal_flip=True,
      fill_mode='nearest')'''

train_gen = datagen.flow_from_directory(
    data_dir, class_mode= 'categorical', batch_size= batch_size,target_size= resize,color_mode= 'grayscale'
)

model = Sequential([
    Conv2D(32, (3,3), input_shape=(50,50,1)),
    MaxPool2D(pool_size=(3,3), padding = 'same'),
    #Conv2D(16,(3,3)),
    #MaxPool2D((3,3)),
	Flatten(),
    #Dense(16, activation='relu'),
	Dense(units=29, activation='softmax')
])

model.compile(optimizer= Adam(lr = 0.001), loss = categorical_crossentropy, metrics= ['accuracy'])
history = model.fit(train_gen, epochs = 20 , shuffle = True)

acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)


plt.show()