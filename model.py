from keras.models import Model
from keras.layers import (Activation, Dropout, AveragePooling2D, Input,
                         Flatten, MaxPooling2D, Convolution2D)
from firemodule import FireModule


class SqueezeNetBuilder:
  def __init__(self, fst_conv_size = 7, avg_pool_size = (2, 2), use_bypasses = False):
    """
      NOTE: for cifar dataset use avg_pool_size = (2, 2), for imagenet, (13, 13).
      NOTE: it is always good to check the model.summary()
            for more detailed information.
            Prevent filters of type (None, `n`, 0, 0). Zeros are forbiden,
            they will cause errors and the model won't compile.
            To prevent it, tune the avg_pool_size.
    """
    self.fst_conv_size = fst_conv_size
    self.avg_pool_size = avg_pool_size
    self.use_bypasses  = use_bypasses

  def __call__(self, input_data, num_of_cls):
    inputs = Input(input_data.shape[1:])

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
    apool   = AveragePooling2D(self.avg_pool_size)(conv_2)

    flatten = Flatten()(apool)
    outputs = Activation("softmax")(flatten)

    return Model(input = inputs, output = outputs)


