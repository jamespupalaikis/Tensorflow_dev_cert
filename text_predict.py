import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('models/sw_predict.h5')

seed_text = "I've got"
next_words = 25
################################################
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

################################################################
for _ in range(next_words):
    token_list = tok.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tok.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)