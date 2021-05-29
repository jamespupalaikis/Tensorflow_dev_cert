import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
tok = Tokenizer(char_level= False)

data = open('Data/SW_EpisodeIV.txt').read()
data2 = open('Data/SW_EpisodeV.txt').read()
data3 = open('Data/SW_EpisodeVI.txt').read()
data = data + data2 + data3

txt = list(filter(lambda x:   (x != 'dialogue') and (x != ' '),data.split('"')[::3]))


print((txt))

tok.fit_on_texts(txt)
total_words = len(tok.word_index) + 1
print(tok.word_index)
print(total_words)

input_sequences = []
for line in txt:
	token_list = tok.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

#print(len(input_sequences), input_sequences)

#now to pad sequences, make predictors + label

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen= max_sequence_len, padding= 'pre'))

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes= total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add((GRU(128, return_sequences= True)))
#model.add((GRU(128, return_sequences= True)))
model.add((GRU(128)))
model.add(Dense(128))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=50)
#print model.summary()
print(model)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plot_graphs(history, 'accuracy')

  plt.show()

model.save('models/sw_predict.h5')