import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split as tts

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Activation,Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy

import math

(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()#label_mode = 'coarse')

print(len(x_train), len(y_train), len(x_test), len(y_test))
print(y_train[0]) #(50000, 32, 32, 3)
#x_train, x_test = np.expand_dims(x_train, axis = 1), np.expand_dims(x_test, axis = 1)
model = Sequential([
    Conv2D(filters = 64, kernel_size= (3,3), input_shape= (32,32,3), activation = 'relu', padding= 'same'),
    BatchNormalization(),
    MaxPool2D(pool_size= (3,3), strides=(3), padding= 'same'),
    Dropout(0.25),
    Conv2D(filters = 64, kernel_size= (3,3),  activation = 'relu', padding= 'same'),
    #BatchNormalization(),
    MaxPool2D(pool_size= (3,3), strides=(3), padding= 'same'),
    Dropout(0.25),
    Conv2D(filters = 64, kernel_size= (3,3),  activation = 'relu', padding= 'same'),
    MaxPool2D(pool_size= (3,3), strides=(3), padding= 'same'),
    Conv2D(filters = 64, kernel_size= (3,3),  activation = 'relu', padding= 'same'),
    MaxPool2D(pool_size= (3,3), strides=(3), padding= 'same'),
    Flatten(),
    Dense(units = 64, activation= 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units=(10), activation=('softmax'))
])

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', min_delta= 0.01, patience=10, restore_best_weights= True, verbose = 2),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta = 0.05, patience=5, min_lr=0.0001, verbose=2)
]


model.compile(optimizer= Adam(lr = 0.001), metrics = 'accuracy', loss = sparse_categorical_crossentropy)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks = my_callbacks, epochs= 30, shuffle = True, batch_size= 100)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.show()