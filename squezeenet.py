from keras.models import Model
from keras.layers import (Input, Dense, Convolution2D, MaxPooling2D, 
                          Dropout, BatchNormalization, Flatten, Merge)
from keras.optimizers import RMSProp
from keras.utils import np_utils

import numpy as np
import theano as tn
import multiprocessing as mp

tn.config.openmp = True
OMP_NUM_THREADS = mp.cpu_count()

class FireModule:
  def __init__(self):
    pass

  def __call__(self, data):
    # data = ...(data)
    # data = ...(data)
    # data = ...(data)
    # data = ...(data)
    pass

    return data

