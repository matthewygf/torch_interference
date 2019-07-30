import tensorflow as tf
# pytorch vision direct translate.

configs = {
  '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(tf.keras.Model):
  def __init__(self, name, config, batch_norm, 
               num_classes=10, data_format='channels_first', input_shape=None):
    super(VGG, self).__init__(name='vgg'+name)
    self.features = make_layers(configs[config], batch_norm=batch_norm, data_format=data_format)
    self.num_classes=num_classes
    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Flatten(name='flatten'),
      tf.keras.layers.Dense(4096, activation='relu', name='fc1'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4096, activation='relu', name='fc2'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(num_classes, name='logits')
    ])

  def call(self, inputs):
    x = self.features(inputs)
    x = self.classifier(x)
    return x

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0]
    return tf.TensorShape(batch_size, self.num_classes)

def make_layers(config, batch_norm=False, input_channels=3, data_format='channels_first'):
  bn_axis = 1 if data_format == 'channels_first' else -1
  layers = tf.keras.Sequential()
  for v in config:
    if v == 'M':
      layers.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, data_format=data_format))
    else:
      conv2d = tf.keras.layers.Conv2D(v, 3, data_format=data_format, padding='same')
      if batch_norm:
        layers.add(conv2d)
        layers.add(tf.keras.layers.BatchNormalization(axis=bn_axis))
        layers.add(tf.keras.layers.ReLU())
      else:
        layers.add(conv2d)
        layers.add(tf.keras.layers.ReLU())

  return layers

def vgg19(**kwargs):
  return VGG('19', '19', False, **kwargs)

def vgg16(**kwargs):
  return VGG('16', '16', False, **kwargs)