from keras.models import Model
from keras.layers import (Activation, Dropout, AveragePooling2D, Input,
                         Flatten, MaxPooling2D, Convolution2D, merge)

#--------------------------------------------
import theano as tn

tn.config.floatX = "float32"
tn.config.openmp = True
tn.config.openmp_elemwise_minsize = 200000
OMP_NUM_THREADS = 4
#--------------------------------------------

class FireModule:
  """
    The FireModule class mimics the building block of the
    Squeeze Network architecture, described in the paper,
    using Keras's Functional API conventions.
  """
  def __init__(self, squeeze_size, expand_size, stride = 1):
    self.sqz_size  = squeeze_size
    self.expn_size = expand_size
    self.stride    = (stride, stride)


  def __call__(self, data):
    # squeeze layer
    sqz_layer = Convolution2D(self.sqz_size, 1, 1, border_mode = "same",
                            subsample = self.stride, activation = "relu")(data)
    
    # expand layer
    conv_1x1 = Convolution2D(self.expn_size, 1, 1, border_mode = "same",
                            subsample = self.stride, activation = "relu")(sqz_layer)
    conv_3x3 = Convolution2D(self.expn_size, 3, 3, border_mode = "same",
                            subsample = self.stride, activation = "relu")(sqz_layer)

    return merge([conv_1x1, conv_3x3], mode = "concat", concat_axis = 1)


class Placeholder:
  """
    Just a placeholder for DenseSubnet parameter
    in SqueezeNetBuilder class
  """
  def __call__(self, layer):
  	pass

class SqueezeNetBuilder:
  def __init__(self, fst_conv_size = 7, avg_pool_size = (2, 2),
  	                 use_bypasses = False, use_noise = False,
  	                 DenseSubnet = Placeholder):
    """
      NOTE: for cifar dataset use avg_pool_size = (2, 2),
            for imagenet, (13, 13).
      NOTE: it is always good to check the model.summary()
            for more detailed information.
            Prevent filters of type (None, `n`, 0, 0). Zeros are forbidden,
            they will cause errors and the model won't compile.
            To prevent it, tune the avg_pool_size.
    """
    self.fst_conv_size = fst_conv_size
    self.avg_pool_size = avg_pool_size
    self.use_bypasses  = use_bypasses
    self.use_noise     = use_noise
    self.DenseSubnet   = DenseSubnet # either Placeholder (the default value) or
                                     # a special subnet, implemented by the user,
                                     # using the same Functional API
                                     # as the rest of the networks modules.
                                     # A good example is the FireModule class


  def __call__(self, input_data_shape, num_of_cls):
    inputs = Input(input_data_shape)

    conv_1  = Convolution2D(96, self.fst_conv_size, self.fst_conv_size)(inputs)
    mpool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)

    fire_1 = FireModule(16, 64)(mpool_1)
    fire_2 = FireModule(16, 64)(fire_1)

    fire_3   = FireModule(32, 128)(fire_2)
    mpool_2  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire_3)
    fire_4   = FireModule(32, 128)(mpool_2)

    fire_5   = FireModule(48, 192)(fire_4)
    fire_6   = FireModule(48, 192)(fire_5)

    fire_7   = FireModule(64, 256)(fire_6)
    mpool_3  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire_7)
    fire_8   = FireModule(64, 256)(mpool_3)

    dropout = Dropout(0.5)(fire_8)
    conv_2  = Convolution2D(num_of_cls, 1, 1)(dropout)
    # The size should match the output of conv10
    apool   = AveragePooling2D(self.avg_pool_size)(conv_2)

    flatten = Flatten()(apool)
    outputs = Activation("softmax")(
    	                            self.DenseSubnet()(flatten) or flatten
    	                           )

    return Model(input = inputs, output = outputs)

