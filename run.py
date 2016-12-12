from keras.datasets import cifar10, mnist
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from models import SqueezeNetBuilder


datasets = {
#  "cifar100": cifar100,
  "cifar10": cifar10
}

(x_train, y_train), (x_test, y_test) = datasets["cifar10"].load_data()

x_train = x_train.astype("float32") / 255.
x_test  = x_test.astype("float32") / 255.

augumented = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True
                               )
augumented.fit(x_train)

y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

model = SqueezeNetBuilder(7, use_batch_norm = True)(x_train.shape[1:], 10)

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam", metrics = ["accuracy"])

model.fit_generator(augumented.flow(x_train, y_train),
                    samples_per_epoch = x_train.shape[0],
                    nb_epoch = 20
                   )

model.predict(x_test, y_test)
model.save("squeezenet.dump")
