import tensorflow as tf
import pandas as pd
import numpy as np



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Activation,Flatten, BatchNormalization, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy,BinaryCrossentropy

import math


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()



#print(y_train)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

'''tokenizer = Tokenizer(num_words=  10000, oov_token= '<OOV>' ) #ITS ALREADY TOKENIZED!
tokenizer.fit_on_texts(x_train)
windex = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(x_train)'''
paddedseq = pad_sequences(x_train, maxlen = 150, truncating= 'post')

#test_sequences = tokenizer.texts_to_sequences(x_test)
test_paddedseq = pad_sequences(x_test, maxlen = 150)
vocab = 88585
emd_dim = 8
max_length = 150
model = Sequential([
    Embedding(vocab, emd_dim, input_length=max_length ),
    Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)),
    Bidirectional(tf.keras.layers.LSTM(14)),
    #Flatten(),
    Dense(units = 8, activation='relu'),
    Dense(units = 1, activation='sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr = 0.001), metrics = ['accuracy'])
history = model.fit(paddedseq, y_train, batch_size=100, validation_data=(test_paddedseq, y_test), shuffle=True, epochs = 10)

#print(np.unique(paddedseq.flatten()))