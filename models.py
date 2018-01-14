from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import (Activation, Dropout, GlobalAveragePooling2D, Input,
                         Flatten, MaxPooling2D, Conv2D, merge,
                         BatchNormalization)

class FireModule:
  """
    The FireModule class mimics the building block of the
    Squeeze Network architecture, described in the paper,
    using Keras's Functional API conventions.
  """
  def __init__(self, squeeze_size, expand_size, use_batch_norm=False, stride=1):
    self.sqz_size       = squeeze_size
    self.expn_size      = expand_size
    self.use_batch_norm = use_batch_norm
    self.stride         = (stride, stride)


  def __call__(self, data):
    # squeeze layer
    sqz_layer = Conv2D(self.sqz_size, (1, 1), padding="same",
                       strides=self.stride, activation="relu")(data)
    
    # expand layer
    conv_1x1 = Conv2D(self.expn_size, (1, 1), padding="same",
                      strides=self.stride, activation="relu")(sqz_layer)
    conv_3x3 = Conv2D(self.expn_size, (3, 3), padding="same",
                      strides=self.stride, activation="relu")(sqz_layer)

    dump = Concatenate(axis=1)([conv_1x1, conv_3x3])
    return dump if not self.use_batch_norm else BatchNormalization()(dump)


class Placeholder:
  """
    Just a placeholder for DenseSubnet parameter
    in SqueezeNetBuilder class
  """
  def __call__(self, layer):
    pass

class SqueezeNetBuilder:
  def __init__(self, fst_conv_size, use_bypasses=False,
                     use_noise=False, use_batch_norm=True,
                     dropout_prob=0.5,
                     DenseSubnet=Placeholder):
    """
      NOTE: it is always good to check the model.summary()
            for more detailed information.
            Prevent filters of type (None, `n`, 0, 0). Zeros are forbidden,
            they will cause errors and the model won't compile.
            To prevent it, tune the avg_pool_size.
    """
    self.fst_conv_size  = fst_conv_size
    self.dropout_prob   = dropout_prob
    self.use_bypasses   = use_bypasses
    self.use_noise      = use_noise
    self.use_batch_norm = use_batch_norm
    self.DenseSubnet    = DenseSubnet # either Placeholder (the default value) or
                                      # a special subnet, implemented by the user,
                                      # using the same Functional API
                                      # as the rest of the networks modules.
                                      # A good example is the FireModule class


  def __call__(self, input_data_shape, num_of_cls):
    inputs = Input(input_data_shape)

    conv_1  = Conv2D(96, (self.fst_conv_size, self.fst_conv_size))(inputs)
    if self.use_batch_norm:
      conv_1 = BatchNormalization()(conv_1)
    mpool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)

    fire_1 = FireModule(16, 64, self.use_batch_norm)(mpool_1)
    fire_2 = FireModule(16, 64, self.use_batch_norm)(fire_1)

    fire_3   = FireModule(32, 128, self.use_batch_norm)(fire_2 if not self.use_bypasses else Concatenate(axis=1)([fire_1, fire_2]))
    mpool_2  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire_3)
    fire_4   = FireModule(32, 128, self.use_batch_norm)(mpool_2)

    fire_5   = FireModule(48, 192, self.use_batch_norm)(fire_4 if not self.use_bypasses else Concatenate(axis=1)([mpool_2, fire_4]))
    fire_6   = FireModule(48, 192, self.use_batch_norm)(fire_5)

    fire_7   = FireModule(64, 256, self.use_batch_norm)(fire_6 if not self.use_bypasses else Concatenate(axis=1)([fire_5, fire_6]))
    mpool_3  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire_7)
    fire_8   = FireModule(64, 256, self.use_batch_norm)(mpool_3)

    dropout = Dropout(0.5)(fire_8 if not self.use_bypasses else Concatenate(axis=1)([mpool_3, fire_8]))
    conv_2  = Conv2D(num_of_cls, (1, 1))(dropout)
    gapool  = GlobalAveragePooling2D()(conv_2)

    outputs = Activation("softmax")(self.DenseSubnet()(gapool) or gapool)

    return Model(inputs=inputs, outputs=outputs)
