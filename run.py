from keras.models import Model, Sequential
from keras.layers import (Activation, Dropout, AveragePooling2D, Input,
                         Flatten, MaxPooling2D, Convolution2D)
from firemodule import FireModule
from keras.datasets import cifar10, mnist
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np


datasets = {
  "mnist": mnist,
  "cifar": cifar10
}

(x_train, y_train), (x_test, y_test) = datasets["cifar"].load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


inputs = Input(x_train.shape[1:])

layer = Convolution2D(96, 7, 7)(inputs)
layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

layer = FireModule(16, 64)(layer)
layer = FireModule(16, 64)(layer)

layer = FireModule(32, 128)(layer)
layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)
layer = FireModule(32, 128)(layer)

layer = FireModule(48, 192)(layer)
layer = FireModule(48, 192)(layer)

layer = FireModule(64, 256)(layer)
layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)
layer = FireModule(64, 256)(layer)

layer = Dropout(0.5)(layer)
layer = Convolution2D(10, 1, 1)(layer)
layer = AveragePooling2D((2, 2))(layer)

layer = Flatten()(layer)
layer = Activation("softmax")(layer)
model = Model(input = inputs, output = layer)

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop", metrics = ["accuracy"])
model.fit(x_train, y_train)
model.predict(x_test, y_test)
model.save("squeezenet.dump")
