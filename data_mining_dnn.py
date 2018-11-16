# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import keras
from keras import Sequential
from keras.layers import Dense, Dropout

df = pd.read_csv('googleplaystore_clean_val.csv')
a = np.copy(df.values)
a = np.delete(a, 0, axis=1)

y = np.copy(a[:, 0])
x = np.delete(a, 0, axis=1)
print(x.shape)

x_train = x[0:7000]
y_train = y[0:7000]
x_test = x[7000:8196]
y_test = y[7000:8196]
print(x_train.shape)
print(x_test.shape)

# x.max(axis=0)
# for i in x_train:
#     print(i)
#
# print(x.max(axis=0))

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std

# print(x_train)
# print(y_train)

# for i in x_train:
#     print(i)


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(9,)))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model


if __name__ == '__main__':
    # model = build_model()
    # model.fit(x_train, y_train, epochs=80, batch_size=16, verbose=1)
    # model.save('dnn2.h5')
    model = keras.models.load_model('dnn.h5')
    y_ = model.predict(x_test).reshape(-1)
    print(np.abs(y_ - y_test))
    # print(y_test)
