'''
Direct translation from pytorch, as a comparison.
'''
import tensorflow as tf
from tensorflow.keras import layers, Model
from tf_image_models.common_ops import Conv2D_Pad


class BasicBlock(layers.Layer):
  expansion = 1
  def __init__(self, name, planes, 
               stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None,
               data_format='channels_first'):
    super(BasicBlock, self).__init__(name='basic_block_'+name)
    self.bn_axis = 1 if data_format == 'channels_first' else -1
    self.stride=stride
    self.downsample=downsample
    
    if norm_layer is None:
      norm_layer = layers.BatchNormalization
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
    
    # both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = Conv2D_Pad(planes, 3, stride, data_format=data_format, use_bias=False)
    self.bn1 = norm_layer(axis=self.bn_axis)
    self.relu = layers.ReLU()
    self.conv2 = Conv2D_Pad(planes, 3, 1, data_format=data_format, use_bias=False)
    self.bn2 = norm_layer(axis=self.bn_axis)


  def call(self, inputs):
    identity = inputs

    out = self.conv1(inputs)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      tf.compat.v1.logging.info("downsample identity at layer %s", self.name)
      identity = self.downsample(inputs)
      
    out += identity
    out = self.relu(out)
    return out
  
class Bottleneck(layers.Layer):
  expansion = 4

  def __init__(self, name, planes, 
               stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None, 
               data_format='channels_first'):
    super(Bottleneck, self).__init__(name='bottleneck_'+name)
    self.bn_axis = 1 if data_format == 'channels_first' else -1
    self.downsample = downsample
    self.strides = stride
    self.data_format = data_format

    if norm_layer is None:
      norm_layer= layers.BatchNormalization
    
    width = int(planes * (base_width / 64.)) * groups
    # both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = Conv2D_Pad(width, 1, 1, use_bias=False, data_format=data_format, )
    self.bn1 = norm_layer(axis=self.bn_axis)
    self.group_convs = []
    self.groups = groups
    if groups == 1:
      self.per_group_c = width
      self.group_convs.append(Conv2D_Pad(self.per_group_c, 3, stride, data_format=data_format, use_bias=False, name='group_conv_only'))
    else:
      self.per_group_c = width // groups
      for i in range(groups):
        self.group_convs.append(Conv2D_Pad(self.per_group_c, 3, stride, data_format=data_format, name='group_conv'+str(i)))
    
    self.bn2 = norm_layer(axis=self.bn_axis)
    self.conv3 = Conv2D_Pad(planes * self.expansion, 1, 1, use_bias=False, data_format=data_format)
    self.bn3 = norm_layer(axis=self.bn_axis)
    self.relu = layers.ReLU()

  def call(self, inputs):
    identity = inputs

    out = self.conv1(inputs)
    out = self.bn1(out)
    out = self.relu(out)

    # group convs
    if self.groups > 1 :
      outs = []
      for i in range(self.groups):
        # split output channels into groups
        if self.data_format == 'channels_first':
          g_out = layers.Lambda(lambda x: x[:, self.per_group_c*i:self.per_group_c*i+self.per_group_c, :, :])(out)
        else:
          g_out = layers.Lambda(lambda x: x[:, :, :, self.per_group_c*i:self.per_group_c*i+self.per_group_c])(out)
        outs.append(self.group_convs[i](g_out))

      out = layers.Concatenate(name='grouped')(outs)
    else:
      out = self.group_convs[0](out)
      
    out = self.bn2(out)
    out = self.relu(out)

    # expansion
    out = self.conv3(out)
    out = self.bn3(out)
    
    if self.downsample is not None:
      identity = self.downsample(inputs)
    
    out += identity
    out = self.relu(out)
    return out

class ResNet(Model):
  def __init__(self, name, block, configs, num_classes=10,
               data_format='channels_first', input_shape=None, 
               groups=1, width_per_group=64, replace_stride_with_dilation=None,
               norm_layer=None):
    super(ResNet, self).__init__(name='resnet'+name)
    self.bn_axis = 1 if data_format == 'channels_first' else -1
    self.data_format = data_format
    if norm_layer is None:
      tf.compat.v1.logging.info("using batchnorm as the default normalization layer ")
      norm_layer = layers.BatchNormalization
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError('replace stride with dilation should be None'
                       'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    self.padd1 = layers.ZeroPadding2D(padding=((3,3), (3,3)), data_format=data_format, input_shape=input_shape)
    self.conv1 = Conv2D_Pad(self.inplanes, 7, strides=2, use_bias=False, data_format=data_format)
    if isinstance(self._norm_layer, layers.BatchNormalization):
      self.n1 = self._norm_layer(axis=self.bn_axis)
    else:
      self.n1 = self._norm_layer()
    self.relu = layers.ReLU()
    self.padd2 = layers.ZeroPadding2D(padding=((1,1), (1,1)), data_format=self.data_format)
    self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, data_format=data_format)
    self.avg_pool = layers.GlobalAveragePooling2D(data_format=data_format)
    self.classifier = layers.Dense(num_classes, name='logits')
    
    # residual layers
    self.layer1 = self._make_layer('layer1_block_', block, 64, configs[0])
    self.layer2 = self._make_layer('layer2_block_', block, 128, configs[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer('layer3_block_', block, 256, configs[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer('layer4_block_', block, 512, configs[3], stride=2, dilate=replace_stride_with_dilation[2])


  def _make_layer(self, name, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = tf.keras.Sequential([
        Conv2D_Pad(planes * block.expansion, 1, 
                      strides=stride, data_format=self.data_format),
        self._norm_layer(axis=self.bn_axis)
      ])
    se_layers = tf.keras.Sequential()
    se_layers.add(
      block(name+'0', planes, stride=stride, 
            downsample=downsample, groups=self.groups, 
            base_width=self.base_width, dilation=previous_dilation, 
            data_format=self.data_format))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      se_layers.add(block(name+str(_), planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, data_format=self.data_format))
    return se_layers
  
  def call(self, inputs):
    x = self.padd1(inputs)
    x = self.conv1(x)
    x = self.n1(x)
    x = self.relu(x)
    x = self.padd2(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avg_pool(x)
    x = self.classifier(x)
    return x
  
def resnet18(**kwargs):
  return ResNet('18', BasicBlock, [2,2,2,2], **kwargs)

def resnet50(**kwargs):
  return ResNet('50', Bottleneck, [3,4,6,3], **kwargs)




    


