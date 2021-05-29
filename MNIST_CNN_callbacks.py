import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = np.expand_dims(train_x, axis = 3)
test_x = np.expand_dims(test_x, axis = 3)
print(train_x.shape) #(60000, 28, 28, 1)
model = Sequential([
    Conv2D(filters=16, kernel_size= (2,2), activation= 'relu', input_shape=(28,28,1), padding = 'same'),
    MaxPool2D(pool_size= (2,2), strides = 2),
    Flatten(),
    Dense(units = 10, activation = 'softmax')
])

model.compile(optimizer= Adam(learning_rate = 0.001), loss = sparse_categorical_crossentropy, metrics = ['accuracy'])



#Callbacks: https://keras.io/api/callbacks/
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', min_delta= 0.01, patience=4, restore_best_weights= True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta = 0.05, patience=3, min_lr=0.0001)
]

history = model.fit(train_x,train_y, validation_data= (test_x,test_y), epochs = 5, shuffle = True, batch_size=10, callbacks = my_callbacks)

print(history.history['val_accuracy'])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.show()
#TODO: plot history, add callbacks, maybe make the model a bit better