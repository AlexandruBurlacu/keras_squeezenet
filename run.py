#!squeeze/bin/python
from keras.datasets import cifar10
from eve import Eve
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from models import SqueezeNetBuilder

import argparse

config = {
  "description" : """CLI to run the slighlty enhanced SqueezeNet model with Eve optimizer.
                   All parameters have default values.""",
  "epilog" : "It is not the finished version, there's still functionality to be implemented."
}

BN_HELP = "Parameter to switch on/off batch normalization."
BY_HELP = "Parameter to switch on/off usage of bypasses."
NS_HELP = "Parameter to switch on/off usage of Gaussian Noise."
CONV_HELP = """It should be choosed in accordance to the dataset used.
             Currently it is recomended not to tune this parameter."""

parser = argparse.ArgumentParser(**config)
parser.add_argument("-bn", "--batch_norm", default = True, type = bool, help = BN_HELP)
parser.add_argument("-by", "--bypasses", default = True, type = bool, help = BY_HELP)
parser.add_argument("-ns", "--noise", default = False, type = bool, help = NS_HELP)
parser.add_argument("--fst_conv", default = 7, type = int, choices = range(1, 13), help = CONV_HELP)
args = parser.parse_args()


datasets = {
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

model = SqueezeNetBuilder(fst_conv_size = args.fst_conv,
	                      use_batch_norm = args.batch_norm,
	                      use_bypasses = args.bypasses,
	                      use_noise = args.noise)(x_train.shape[1:], 10)

eve = Eve()

model.compile(loss = "categorical_crossentropy",
              optimizer = eve, metrics = ["accuracy"])

model.fit_generator(augumented.flow(x_train, y_train),
                    samples_per_epoch = x_train.shape[0],
                    nb_epoch = 20
                   )

model.predict(x_test, y_test)
model.save("squeezenet.dump")
