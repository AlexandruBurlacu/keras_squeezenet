from keras.models import Model
from keras.layers import (Input, Dense, Convolution2D, MaxPooling2D, 
                          Dropout, BatchNormalization, Flatten, merge)
from keras.optimizers import RMSProp
from keras.utils import np_utils

import numpy as np
import theano as tn
import multiprocessing as mp

tn.config.openmp = True
OMP_NUM_THREADS = mp.cpu_count()

class FireModule:
  def __init__(self, squeeze_size, expand_size, stride = 1):
    self.sqz_size = squeeze_size
    self.expn_size = expand_size
    self.stride = (stride, stride)

  def __call__(self, data):
    # squeeze layer
    sqz_layer = Convolution2D(self.sqz_size, 1, 1,
                            subsample = self.stride, activation = "relu")(data)
    
    # expand layer
    conv_1x1 = Convolution2D(self.expn_size, 1, 1,
                            subsample = self.stride, activation = "relu")(sqz_layer)
    conv_3x3 = Convolution2D(self.expn_size, 3, 3,
                            subsample = self.stride, activation = "relu")(sqz_layer)

    return merge([conv_1x1, conv_3x3], mode = "concat", concat_axis = 1)

