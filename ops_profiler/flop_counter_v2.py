import torch
#http://studyai.com/article/a718990b
def calculate_FLOPs_scale(model, input_size, multiply_adds=False, use_gpu=False):
    """
    forked from FishNet @ github
    https://github.com/kevin-ssy/FishNet/blob/master/utils/profile.py
    another: https://github.com/Lyken17/pytorch-OpCounter
    no bias: K^2 * IO * HW
    multiply_adds : False in FishNet Paper, but True in DenseNet paper
    """
    assert isinstance(model, torch.nn.Module)
    USE_GPU = use_gpu and torch.cuda.is_available()

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)

    def deconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_deconv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(deconv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
    multiply_adds = multiply_adds
    list_conv, list_deconv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], [], []
    foo(model)
    input = torch.rand(input_size)
    if USE_GPU:
        input = input.cuda()
        model = model.cuda()
    _ = model(input)
    total_flops = (sum(list_conv) + sum(list_deconv) + sum(list_linear)
                   + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('  + Number of FLOPs: %.5fG' % (total_flops / 1e9 / 2))