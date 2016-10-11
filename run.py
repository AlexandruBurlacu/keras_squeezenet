from keras.datasets import cifar10, mnist
from keras.optimizers import SGD
from keras.utils import np_utils

from model import SqueezeNetBuilder

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

model = SqueezeNetBuilder()(x_train, 10)

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop", metrics = ["accuracy"])
model.fit(x_train, y_train)
model.predict(x_test, y_test)
model.save("squeezenet.dump")
