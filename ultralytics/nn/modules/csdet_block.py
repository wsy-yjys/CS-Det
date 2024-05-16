"""
CS-Det Block modules
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

__all__ = ('IEPR', 'conv_bn', 'RepBlock3')


def get_activation(name='silu', inplace=True):
    if name is None or name==False or name == 'identity':
        return nn.Identity()
    elif name==True:
        name = 'silu'

    if isinstance(name, str):
        name = name.lower()
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'gelu':
            module = nn.GELU()
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'relu6':
            module = nn.ReLU6(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'hardswish':
            module = nn.Hardswish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


class conv_bn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 groups=1,
                 bias=False):

        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        return self.bn(self.conv(x))

    def forward_fuse(self,x):
        return self.conv(x)

    def switch_to_deploy(self):
        kernel = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel_fused, bias_fused = kernel * t, (bias - running_mean) * gamma / std + beta
        # Reparameterization
        self.conv.weight.data = kernel_fused
        setattr(self.conv, 'bias', torch.nn.Parameter(bias_fused,requires_grad=False))
        self.__delattr__('bn')
        self.forward = self.forward_fuse


class FFN(nn.Module):
    def __init__(self, c1=3,
                 c2=3,
                 act="SiLU",
                 e=2,
                 shortcut=True):

        super().__init__()
        self.m = nn.Sequential(
                    conv_bn(c1, int(e * c2), 1, 1, 0),
                    get_activation(act),
                    conv_bn(int(e * c2), c2, 1, 1, 0),
                )
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.m(x) if self.shortcut else self.m(x)

    def switch_to_deploy(self):
        for m in self.m:
            if isinstance(m, (conv_bn,)):
                m.switch_to_deploy()


class Partial_conv3(nn.Module):
    def __init__(self,
                 dim,
                 n_div=2,
                 forward='split_cat',
                 act=False,
                 bn=True,
                 type="normal",
                 k=3,
                 d=1,
                 groups=1,
                 deepextend=4,
                 shortcut=False):       # Pconvshortcut: RepBlock3 use shortcut

        super().__init__()
        self.dim_conv3 = int(dim // n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.type = type

        if type=="normal":
            self.partial_conv3 = Conv(self.dim_conv3, self.dim_conv3, k, 1, act=act, d=d, g=groups) if bn else nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k//2, bias=False,dilation=d)
        elif type=="deepextend":
            self.partial_conv3 = RepBlock3(self.dim_conv3, self.dim_conv3, e=deepextend, groups=groups, shortcut=shortcut)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv3(x1), x2), 1)

    def forward_slicing(self, x):
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def switch_to_deploy(self):
        self.forward = self.forward_slicing
        # conv_bn deploy
        if isinstance(self.partial_conv3, (conv_bn,RepBlock3)):
            self.partial_conv3.switch_to_deploy()

class RepBlock3(nn.Module):
    """conv1x1-conv3x3-conv1x1"""
    def __init__(self,
                 c1=3,
                 c2=512,
                 k=(1, 3, 1),
                 stride=(1, 1, 1),
                 groups=1,
                 act="silu",
                 finalact=True,
                 e=1,
                 shortcut=False,
                 ):

        super().__init__()
        self.groups = groups
        self.kernel = k
        self.pad_pixels = k[1]//2
        self.add = shortcut and c1 == c2 and stride[1]==1
        self.finalact = finalact     # Whether to use the activation function at the end
        self.cv1 = conv_bn(in_channels=c1, out_channels=c2 * e, kernel_size=k[0], stride=stride[0], padding=k[0] // 2, groups=1, bias=False)
        self.cv2 = conv_bn(in_channels=c2 * e, out_channels=c2 * e, kernel_size=k[1], stride=stride[1], padding=0, groups=groups, bias=False)
        self.cv3 = conv_bn(in_channels=c2 * e, out_channels=c2, kernel_size=k[2], stride=stride[2], padding=k[2] // 2, groups=1, bias=False)

        # rbr_identity
        if self.add:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1)
        if finalact:
            self.activation = get_activation(act)

    # deploy method（as a single block）
    def forward_fuse_finalact(self, x):
        return self.activation(self.cv(x))

    # deploy method（as a block in PC2f_Decay）
    def forward_fuse(self, x):
        return self.cv(x)

    def forward(self,x):
        if self.add:
            x_identity = x.clone()
        x = self.cv1(x)
        if self.kernel[1]>=3:
            x_pad = self.padlayer(x, self._fuse_bn_tensor(self.cv1.bn)[1])    # wo bias
            # x_pad = self.padlayer(x, self._fuse_bn_tensor(self.cv1)[1])     # with bias
        x = self.cv2(x_pad)
        x = self.cv3(x)
        x = self.rbr_identity(x_identity) + x if self.add else x
        if self.finalact:       # normal act
            x = self.activation(x)
        return x

    def padlayer(self, x ,pad_values):
        x = F.pad(x, [self.pad_pixels] * 4)
        # pad_pixels=K//2
        pad_values = pad_values.view(1, -1, 1, 1)
        x[:, :, 0:self.pad_pixels, :] = pad_values
        x[:, :, -self.pad_pixels:, :] = pad_values
        x[:, :, :, 0:self.pad_pixels] = pad_values
        x[:, :, :, -self.pad_pixels:] = pad_values
        return x

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.cv1)
        self.group2vanilla()
        kernel2, bias2 = self._fuse_bn_tensor(self.cv2)
        kernel3, bias3 = self._fuse_bn_tensor(self.cv3)
        # Vertically fuse cv1 and cv2
        k = F.conv2d(kernel2, kernel1.permute(1, 0, 2, 3))  # [input, weight]
        b = (kernel2 * bias1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + bias2
        # Vertically fuse cv2 and cv3
        weight_fused = torch.einsum('oi,icjk->ocjk', kernel3.squeeze(3).squeeze(2), k)
        bias_fused = bias3 + (b.view(1, -1, 1, 1) * kernel3).sum(3).sum(2).sum(1)
        # Horizontal fusion cv and BN
        if self.add:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            weight_fused += kernelid
            bias_fused += biasid
        # Reparameterization
        self.cv = nn.Conv2d(in_channels=self.cv1.conv.in_channels, out_channels=self.cv3.conv.out_channels,
                            kernel_size=self.cv2.conv.kernel_size, stride=self.cv2.conv.stride,
                            padding=self.pad_pixels, dilation=self.cv2.conv.dilation,
                            groups=1, bias=True)
        self.cv.weight.data = weight_fused
        self.cv.bias.data = bias_fused
        # Remove excess branches
        self.__delattr__('cv1')
        self.__delattr__('cv2')
        self.__delattr__('cv3')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        # Set inference mode
        if self.finalact:
            self.forward = self.forward_fuse_finalact
        else:
            self.forward = self.forward_fuse

    def group2vanilla(self, ):
        # group conv——>vanilla conv
        kernel = self.cv2.conv.weight.data
        group_out_channels = self.cv2.conv.out_channels // self.cv2.conv.groups     # Output channel for each group
        group_kernel_list = []
        for i in range(self.groups):
            zeros_kernel = torch.zeros([group_out_channels, self.cv2.conv.in_channels, 3, 3]).to(self.cv2.conv.weight.device)
            zeros_kernel[:, group_out_channels * i:group_out_channels * (i + 1), :, :] = kernel[group_out_channels * i:group_out_channels * (i + 1),:, :, :]
            group_kernel_list.append(zeros_kernel)
        # obtained the equivalent weights of conv3x3 after BasicBlock reargument
        weight = torch.cat(group_kernel_list, dim=0)
        self.cv2.conv.weight.data = weight

    def _fuse_bn_tensor(self, branch):
        if hasattr(branch, "conv"):
            kernel = branch.conv.weight
            bias = branch.conv.bias if branch.conv.bias is not None else 0
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = branch.num_features
            kernel_value = np.zeros((branch.num_features, input_dim, self.kernel[1], self.kernel[1]), dtype=np.float32)
            for i in range(branch.num_features):
                kernel_value[i, i % input_dim, (self.kernel[1]-1)//2, (self.kernel[1]-1)//2] = 1     # 中间元素变1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            bias = 0
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, (bias - running_mean) * gamma / std + beta


class IEPR(nn.Module):
    """The IEPR Block(Inverted extended partial residual)"""
    def __init__(self,
                 c1,
                 c2,
                 n_div=2,
                 e=2,               # expansion rate
                 shortcut=True,
                 Pconvtype="normal",
                 groups=1,
                 deepextend=4,
                 act="SiLU",
                 ):

        super().__init__()
        self.shortcut = shortcut
        self.cv = nn.Sequential(
            Conv(c1, int(e * c2), 1, 1, act),
            Partial_conv3(int(e * c2), n_div=n_div, act=act, type=Pconvtype, groups=groups, deepextend=deepextend),
            Conv(int(e * c2), c2, 1, 1, False))

    def forward(self, x):
        return x + self.cv(x) if self.shortcut else self.cv(x)

    def switch_to_deploy(self):
        self.cv[1].switch_to_deploy()

