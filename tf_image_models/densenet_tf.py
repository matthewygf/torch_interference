# modified from pytorch densenet
# so it runs with my script
# NOTE: WIP

import tensorflow as tf

tf.keras.backend.clear_session()

# NOTE: default cifar10 configuration
# 3 dense blocks, 
# 40 depth,
# num Layers first_kernel, second_kernel, stride
blocks_config = [ 
                  (12, 1, 3, 1),
                  (12, 1, 3, 1),
                  (12, 1, 3, 1)
                ]


class DenseNetTF(tf.keras.Model):
  def __init__(self, num_classes=10, is_training=True,
               initial_features=16, initial_kernel_size=3, initial_stride = 1,
               blocks_config=None, growth_rate=12, drop_rate=0, bn_size=4, data_format='channels_first'):

    super(DenseNetTF, self).__init__()
    self.bn_axis = 1 if data_format == 'channels_first' else -1

    if len(blocks_config) > 3:
      # bigger densenet
      self.features = tf.keras.Sequential([
        tf.keras.ZeroPadding2D(padding=3, data_format=data_format, name='feature_zero_pad'),
        tf.keras.layers.Conv2D(initial_features, initial_kernel_size, initial_stride, data_format=data_format, use_bias=False, kernel_initializer='he_normal', name='conv1'),
        tf.keras.layers.BatchNormalization(axis=self.bn_axis),
        tf.keras.layers.ReLU(),
        tf.keras.ZeroPadding2D(padding=1, data_format=data_format, name='pool1_zero_pad'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
      ])
    else:
      self.features = tf.keras.layers.Conv2D(initial_features, initial_kernel_size, initial_stride, 'same', channels, use_bias=False, kernel_initializer='he_normal')
  



