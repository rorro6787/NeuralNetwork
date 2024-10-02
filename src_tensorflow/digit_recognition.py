# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bpn33gpv-uLPzEkRgP8GXeyON4y4L389
"""

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

import matplotlib.pyplot as plt

image = x_train[0]
plt.imshow(image, cmap='gray')

x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

x_train = x_train / 255
x_valid = x_valid / 255

import tensorflow.keras as keras
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

from tensorflow.keras.models import Sequential

model = Sequential()

from tensorflow.keras.layers import Dense

model.add(Dense(units=512, activation='relu', input_shape=(784,)))

model.add(Dense(units = 512, activation='relu'))

model.add(Dense(units = 10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid)
)

model.save('model.keras')