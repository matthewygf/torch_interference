import tensorflow as tf
from tensorflow.keras import layers, Model
from tf_image_models.common_ops import Conv2D_Pad

class SimpleNet(Model):
  def __init__(self, name, num_classes=10, data_format='channels_last', input_shape=None):
    super(SimpleNet, self).__init__(name=name)
    self.data_format=data_format
    self.conv1 = Conv2D_Pad(16, 3, 2, data_format=data_format, use_bias=False)
    self.bn_axis = 1 if data_format == 'channels_first' else -1
    self.bn = layers.BatchNormalization(axis=self.bn_axis)
    self.relu = layers.ReLU()

    self.dense = layers.Dense(num_classes, name='logits')
  
  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.bn(x)
    x = self.relu(x)
    x = layers.Flatten(data_format=self.data_format)(x)
    print(x.shape)
    x = self.dense(x)
    return x

def debug_model(**kwargs):
  return SimpleNet('debug', **kwargs)