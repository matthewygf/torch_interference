# modified from https://github.com/Lyken17/pytorch-OpCounter
# reason is that i want to count number of convs etc 
# considering torch/vision modules

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from image_models.model import MBConvBlock, SwishActivation
from image_models.utils import Conv2dSamePadding, Pad2d
from .count_hooks import *

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose2d: count_convtranspose2d,
    # Conv2d Padding, should add neglible flop costs
    # to pad the X, as output element is still being calculated here.
    Conv2dSamePadding: count_convNd,
    Pad2d: count_pad2d,
    MBConvBlock: count_mbconv_misc,
    
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,
    SwishActivation: count_swish,

    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,

    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
    nn.LSTM: count_lstm
}

activation_sets = set(['softmax', 'sigmoid', 'relu', 'tan', 'relu6'])

def profile(model, input_size, custom_ops={}, device="cpu", logger=None, is_cnn=False):
    handler_collection = []
    logger.info("start counting for %s", str(model.__class__.__name__))

    def add_hooks(m):
        # https://discuss.pytorch.org/t/use-and-abuse-of-register-buffer/4128/2
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            if len(list(m.children())) > 2:  # skip adding param for non-leaf module
                continue
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            logger.warning("Not implemented for: %s", str(m.__class__.__name__))

        if fn is not None:
            #logger.info("Register FLOP counter for module: %s", str(m.__class__.__name__))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
    
    original_device = model.parameters().__next__().device
    training = model.training

    model.eval().to(device)
    model.apply(add_hooks)
    
    total_convs = 0
    total_linear = 0
    total_activation = 0
    total_rnns = 0
    total_others = 0
    for idx, name_mod in enumerate(model.named_modules()):
        if len(list(name_mod[1].children())) >= 1: continue
        # NOTE: DOES NOT COUNT SKIP_CONNECT
        # NOTE: DOES NOT COUNT ATTENTION
        name = str(name_mod[1].__class__.__name__).lower()
        print(idx ,'-->', name_mod[0])
        print(name)
        if 'conv' in name:
            total_convs += 1
        elif 'linear' in name:
            total_linear += 1
        elif name in activation_sets:
            total_activation +=1 
        elif 'lstm' in name or 'gru' in name:
            total_rnns += 1
        else:
            total_others += 1

    logger.info("Count total num of register modules: %d", len(handler_collection))

    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
      model(x)
      

    total_ops = 0
    total_params = 0
    for m in model.modules():
        total_ops += m.total_ops
        if len(list(m.children())) > 1:   # skip adding param for non-leaf module
              continue
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()
    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    
    return total_ops, total_params, total_convs, total_linear, total_activation, total_others