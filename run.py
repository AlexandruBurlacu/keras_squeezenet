#! squeeze/bin/python

from __future__ import print_function

import argparse

config = {
  "description" : """CLI to run the slighlty enhanced SqueezeNet model with Eve optimizer.
                   All parameters have default values.""",
  "epilog" : "It is not the finished version, there's still functionality to be implemented."
}

DEVICE_HELP = "Choice the device, either 'cpu' or 'gpu'. (default = cpu)"
EP_HELP = "Parameter to set the number of epochs. (default = 25)"
BN_HELP = "Parameter to switch on/off batch normalization. (default = true)"
BY_HELP = "Parameter to switch on/off usage of bypasses. (default = true)"
DO_HELP = "Parameter to switch choice the dropout probability. (default = 0.5)"
NS_HELP = "Parameter to switch on/off usage of Gaussian Noise. (default = false) [Not yet implemented]"
CONV_HELP = """It should be choosed in accordance to the dataset used.
             Currently it is recomended not to tune this parameter. (default = 7)"""

parser = argparse.ArgumentParser(**config)
parser.add_argument("--device", default = 'cpu', type = str, choices = ['cpu', 'gpu'], help = DEVICE_HELP)
parser.add_argument("-bn", "--batch_norm", default = True, type = bool, help = BN_HELP)
parser.add_argument("-ep", "--nb_epochs", default = 25, type = int, help = EP_HELP)
parser.add_argument("-by", "--bypasses", default = True, type = bool, help = BY_HELP)
parser.add_argument("-ns", "--noise", default = False, type = bool, help = NS_HELP)
parser.add_argument("-do", "--dropout", default = 0.5, type = int, choices = [0, 0.2, 0.3, 0.5, 0.7], help = DO_HELP)
parser.add_argument("--fst_conv", default = 7, type = int, choices = range(1, 13), help = CONV_HELP)
args = parser.parse_args()

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------

from keras.datasets import cifar10

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from numpy import concatenate as concat

from models import SqueezeNetBuilder
from eve import Eve

from keras.utils import plot_model

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------

datasets = {
  "cifar10": cifar10
}

(x_train, y_train), (x_test, y_test) = datasets["cifar10"].load_data()

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------

y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

augumented = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1/255.,
                                horizontal_flip=True)
augumented.fit(x_train)

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------


model = SqueezeNetBuilder(fst_conv_size=args.fst_conv,
                          use_batch_norm=args.batch_norm,
                          use_bypasses=args.bypasses,
                          use_noise=args.noise,
                          dropout_prob=args.dropout)(x_train.shape[1:], 10)

eve = Eve()

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit_generator(augumented.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=x_train.shape[0] / 64,
                    epochs=args.nb_epochs)

print(model.evaluate(x_test, y_test, batch_size=64))

model.save("squeezenet.h5")
