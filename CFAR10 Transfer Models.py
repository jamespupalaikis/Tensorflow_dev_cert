import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy, mean_absolute_percentage_error


(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()#label_mode = 'coarse')


def preprocess_data1(x,y):
    xx = tf.keras.applications.resnet.preprocess_input(x)
    yy = tf.keras.utils.to_categorical(y)
    return xx,yy



def resnet1(x_tr, y_tr,x_te,y_te, save = False):
    x_tr, y_tr = preprocess_data1(x_tr, y_tr)
    x_te, y_te = preprocess_data1(x_te, y_te)
    from tensorflow.keras.applications import ResNet50 as RN
    #create and assign input tensor
    input_t = tf.keras.Input(32,32,3)
    mymodel = RN(include_top= False, input_shape= (32,32,3), weights = 'imagenet')
    print(len(mymodel.layers))
    mymodel.summary()
    for layer in mymodel.layers[:143]:
        layer.trainable = False
    model = Sequential()
    model.add(mymodel)
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer=Adam(lr = 0.001), metrics=['accuracy'], loss = categorical_crossentropy)
    history = model.fit(x_tr, y_tr, validation_data=(x_te, y_te), shuffle=True, batch_size=100, epochs=5)

#################################


def preprocess_data2(x,y):
    xx = tf.keras.applications.resnet_v2.preprocess_input(x)
    yy = tf.keras.utils.to_categorical(y)
    return xx,yy



def resnet2(x_tr, y_tr,x_te,y_te, save = False):#dont run this shit, takes fucking ages
    x_tr, y_tr = preprocess_data2(x_tr, y_tr)
    x_te, y_te = preprocess_data2(x_te, y_te)
    from tensorflow.keras.applications import ResNet152V2 as RN
    #create and assign input tensor
    input_t = tf.keras.Input(32,32,3)
    mymodel = RN(include_top= False, input_shape= (32,32,3), weights = 'imagenet')
    print(len(mymodel.layers))
    mymodel.summary()
    for layer in mymodel.layers[:-36]:
        layer.trainable = False
    model = Sequential()
    model.add(mymodel)
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer=Adam(lr = 0.001), metrics=['accuracy'], loss = categorical_crossentropy)
    history = model.fit(x_tr, y_tr, validation_data=(x_te, y_te), shuffle=True, batch_size=100, epochs=20 )

#resnet1(x_train,y_train, x_test, y_test)
#528

