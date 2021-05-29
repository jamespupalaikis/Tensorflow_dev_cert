import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as k

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Flatten, Dense, Dropout, Conv1D, Lambda
from tensorflow.keras.optimizers import Adam


data = pd.read_csv('data/ethdata.csv')#, delimiter=',')
print(float('671.20'))
def volumefn(volstr):
    #print(volstr[:-1])
    num = float(volstr[:-1])
    pred = volstr[-1]
    if(pred == 'M'):
        val = num * 1e6
    elif(pred == 'K'):
        val = num * 1e3
    return val

def pricefn(pricestr):
    pricestr = pricestr.replace(',','')
    return float(pricestr)

data['Vol.'] = data['Vol.'].apply(volumefn)
data['Price'] = data['Price'].apply(pricefn)
data['Open'] = data['Open'].apply(pricefn)
data['High'] = data['High'].apply(pricefn)
data['Low'] = data['Low'].apply(pricefn)
data['series'] = list(zip(data['Price'].to_list(), data['Vol.'].to_list()))

data1 = data['Price'].to_list()
data = data['series'].to_list()

splitratio = 0.9

leng = int(len(data1) * splitratio)
data1_train = data1[:leng]
data1_test = data1[leng:]

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


window_size = 20
batch_size = 1
shuffle_buffer_size = 1000

dataset = windowed_dataset(data, window_size, batch_size,shuffle_buffer_size )
dataset1 = windowed_dataset(data1_train, window_size, batch_size,shuffle_buffer_size )
print(len(data1))
#for element in dataset.as_numpy_iterator():
#    print(element)

model = Sequential([
    Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[20]),
    Bidirectional(GRU(32,return_sequences=True)),# input_shape = (20,)),
    Bidirectional(GRU(32)),
    Dense(32),
    Dense(32),
    Dense(1),
    Lambda(lambda x: x*50)
])

model.compile(loss="mse", optimizer=Adam(lr = 0.0001))
print('compiled')

model.fit(dataset1, shuffle= False, epochs=25)


forecast=[]
for time in range(len(data1) - window_size):
    #print(time,time + window_size)
    print(time, len(data1) - window_size)
    forecast.append(model.predict(data1[time:time + window_size]))#[np.newaxis]

forecast = forecast#[leng-window_size:]
results = np.array(forecast)[:, 0, 0]#


plt.plot(results, label = 'predicted')
plt.plot(data1, label = 'true')
plt.legend()
plt.show()