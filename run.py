#! squeeze/bin/python
import argparse
import theano as tn

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


# if args.device == 'cpu':
#   tn.config.openmp = True
#   tn.config.openmp_elemwise_minsize = 200000
#   OMP_NUM_THREADS = 4
# else:
#   tn.config.floatX = "float32"
#   tn.config.device = "gpu"

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------

from keras.datasets import cifar10

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from numpy import concatenate as concat

from models import SqueezeNetBuilder
from eve import Eve

from keras.utils.visualize_util import plot

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

def wrappedGenerator(batch_size = 32):
  """
    This augumented data generator makes possible
    to tune the size of the batch that goes into the Neural Net.
    Original's ImageDataGenerator batch size was fixed on 32 items per call.
    It's pretty ugly, I know.
  """
  while True:
    data = next(augumented.flow(x_train, y_train))
    condition = batch_size

    while condition > 32:
      batch = next(augumented.flow(x_train, y_train))
      data = concat([data[0], batch[0]]), concat([data[1], batch[1]])
      condition -= 32

    yield data

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------


model = SqueezeNetBuilder(fst_conv_size = args.fst_conv,
                          use_batch_norm = args.batch_norm,
                          use_bypasses = args.bypasses,
                          use_noise = args.noise,
                          dropout_prob = args.dropout)(x_train.shape[1:], 10)

eve = Eve()

model.compile(loss = "categorical_crossentropy",
              optimizer = eve, metrics = ["accuracy"])

model.fit_generator(wrappedGenerator(64),
                    samples_per_epoch = x_train.shape[0],
                    nb_epoch = args.nb_epochs, verbose = 1
                   )

print model.metrics_names
print model.evaluate(x_test, y_test, batch_size = 64)

model.save("squeezenet.dump")

plot(model, to_file = "model.png")
