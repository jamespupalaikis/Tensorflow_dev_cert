import tensorflow.keras as k
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, Flatten, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
from sklearn.model_selection import train_test_split as tts


realdat= pd.read_csv('data/True.csv')
fakedat= pd.read_csv('data/Fake.csv')
realnum = len(realdat['title'].values)
fakenum = len(fakedat['title'].values)



x = np.array(realdat['title'].to_list() + fakedat['title'].to_list())

y = np.array([1 for i in range(realnum)] + [0 for i in range(fakenum)])
x_train, x_test, y_train, y_test = tts(x,y, shuffle= True, random_state= 0)

print(x_train,'apple', x_train[0])

tokenizer = Tokenizer(oov_token= '<OOV>')
tokenizer.fit_on_texts(x)




maxlen = 200
embdim = 8
total_words = len(tokenizer.word_index) + 1

xs_train = tokenizer.texts_to_sequences(x_train)
xs_train = pad_sequences(xs_train, padding = 'post', maxlen= maxlen)

xs_test = tokenizer.texts_to_sequences(x_test)
xs_test = pad_sequences(xs_test, padding = 'post', maxlen= maxlen)

#print(xs_test)
#print(xs_train)

model = Sequential([
    Embedding(input_dim=total_words ,output_dim=embdim, input_length=maxlen ),
    Bidirectional(GRU(16, return_sequences= True)),
    Bidirectional(GRU(16)),
    Flatten(),
    Dense(16, activation= 'relu'),
    Dense(1, activation= 'sigmoid')

])

model.compile(optimizer= Adam(lr = 0.001), metrics=['accuracy'], loss = binary_crossentropy)

#history = model.fit(xs_train, y_train, validation_data=(xs_test, y_test), batch_size= 64, shuffle= True, epochs = 10)
