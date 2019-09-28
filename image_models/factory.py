
from image_models.resnext import *
from image_models.DPN import *
from image_models.pyramidnet import *
import torchvision.models as models


models_factory = {
  'googlenet': models.googlenet,
  'squeezenet1_0': models.squeezenet1_0,
  'mobilenet': models.mobilenet_v2,
  'mobilenet_large': models.mobilenet_v2,
  'shufflenetv2_0_5': models.shufflenet_v2_x0_5,
  'shufflenetv2_1_0': models.shufflenet_v2_x1_0,
  'shufflenetv2_2_0': models.shufflenet_v2_x2_0,
  'resnext11_2x16d': ResNext11_2x16d,
  'resnext20_2x16d': ResNext20_2x16d,
  'resnext20_2x32d': ResNext20_2x32d,
  'resnext11_2x64d': ResNext11_2x64d,
  'resnext29_2x64d': ResNeXt29_2x64d,
  'resnet18': models.resnet18,
  'resnet34': models.resnet34,
  'resnet50': models.resnet50,
  'densenet121': models.densenet121,
  'densenet161': models.densenet161,
  'densenet169': models.densenet169,
  'densenet40': models.DenseNet,
  'vgg11': models.vgg11,
  'vgg11_bn': models.vgg11_bn,
  'vgg19': models.vgg19,
  # new
  'mnasnet0_5': models.mnasnet0_5,
  'mnasnet1_0': models.mnasnet1_0,
  'mnasnet1_3': models.mnasnet1_3,
  'dpn92': DPN92,
  'dpn26': DPN26,
  'dpn26_small': DPN26_small,
  'pyramidnet_48_110': pyramidnet_48_110,
  'pyramidnet_84_66': pyramidnet_84_66,
  'pyramidnet_84_110': pyramidnet_84_110,
  'pyramidnet_270_110_bottleneck': pyramidnet_270_110_bottleneck,
  'resnet_wide_18_2': resnet_wide_18_2
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet_wide_18_2(pretrained=False, **kwargs):
  return models.ResNet(Bottleneck, [2,2,2,2], width_per_group=64 * 2, **kwargs)


def get_model(model_name, dataset_name, dataset_classes):
  model_fn = models_factory[model_name]
  # TODO: use a Dict and update.
  if 'google' in model_name: 
    model = model_fn(pretrained=False, transform_input=False, aux_logits=False, num_classes=dataset_classes)
  elif 'mobilenet_large' in model_name:
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 32, 2, 2],
        [6, 64, 3, 2],
        [6, 96, 4, 2],
        [6, 128, 4, 1],
        [6, 256, 2, 2],
        [6, 512, 2, 1],
    ]
    model = model_fn(pretrained=False, num_classes=dataset_classes, inverted_residual_setting=inverted_residual_setting)
  elif 'efficientnet' in model_name and 'v2' not in model_name:
    model = model_fn(model_name, {'num_classes': dataset_classes})
  elif 'densenet40' in model_name:
    model = model_fn(num_classes=dataset_classes, growth_rate=12, num_init_features=16, block_config=(12,12,12))
  elif 'pyramid' in model_name:
    model = model_fn(dataset=dataset_name, num_classes=dataset_classes)
  else:
    model = model_fn(pretrained=False, num_classes=dataset_classes)
  return model