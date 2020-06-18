#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from VSR.Util.Utility import to_list


class MeanShift(nn.Conv2d):
  def __init__(self, mean_rgb, sub, rgb_range=1.0):
    super(MeanShift, self).__init__(3, 3, 1)
    sign = -1 if sub else 1
    self.weight.data = torch.eye(3).view(3, 3, 1, 1)
    self.bias.data = torch.Tensor(mean_rgb) * sign * rgb_range
    # Freeze the mean shift layer
    for params in self.parameters():
      params.requires_grad = False


class Activation(nn.Module):
  def __init__(self, act, **kwargs):
    super(Activation, self).__init__()
    if act is None:
      self.f = lambda t: t
    if isinstance(act, str):
      self.name = act.lower()
      in_place = kwargs.get('in_place', True)
      if self.name == 'relu':
        self.f = nn.ReLU(in_place)
      elif self.name == 'prelu':
        self.f = nn.PReLU(num_parameters=kwargs.get('num_parameters', 1),
                          init=kwargs.get('init', 0.25))
      elif self.name in ('lrelu', 'leaky', 'leakyrelu'):
        self.f = nn.LeakyReLU(negative_slope=kwargs.get('negative_slope', 1e-2),
                              inplace=in_place)
      elif self.name == 'tanh':
        self.f = nn.Tanh()
      elif self.name == 'sigmoid':
        self.f = nn.Sigmoid()
    elif callable(act):
      self.f = act

  def forward(self, x):
    return self.f(x)


class EasyConv2d(nn.Module):
  """ Convolution maker, to construct commonly used conv block with default
  configurations.

  Support to build Conv2D, ConvTransposed2D, along with selectable normalization
  and activations.
  Support normalization:
  - Batchnorm2D
  - Spectralnorm2D
  Support activation:
  - Relu
  - PRelu
  - LeakyRelu
  - Tanh
  - Sigmoid
  - Customized callable functions

  Args:
      in_channels (int): Number of channels in the input image
      out_channels (int): Number of channels produced by the convolution
      kernel_size (int or tuple): Size of the convolving kernel
      stride (int or tuple, optional): Stride of the convolution. Default: 1
      padding (str, optional): 'same' means $out_size=in_size // stride$ or
                                $out_size=in_size * stride$ (ConvTransposed);
                                'valid' means padding zero.
      dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
      groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
      use_bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
      use_bn (bool, optional): If ``True``, adds Batchnorm2D module to the output.
      use_sn (bool, optional): If ``True``, adds Spectralnorm2D module to the output.
      transposed (bool, optional): If ``True``, use ConvTransposed instead of Conv2D.
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding='same', dilation=1, groups=1, activation=None,
               use_bias=True, use_bn=False, use_sn=False, transposed=False,
               **kwargs):
    super(EasyConv2d, self).__init__()
    padding = padding.lower()
    assert padding in ('same', 'valid')
    if transposed:
      assert padding == 'same'
      q = kernel_size % 2  # output padding
      p = (kernel_size + q - stride) // 2  # padding
      net = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                p, q, groups, use_bias, dilation)]
    else:
      if padding == 'same':
        padding_ = (dilation * (kernel_size - 1) - stride + 2) // 2
      else:
        padding_ = 0
      net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                       padding_, dilation, groups, use_bias)]
    if use_sn:
      net[0] = nn.utils.spectral_norm(net[0])
    if use_bn:
      net += [nn.BatchNorm2d(
          out_channels,
          eps=kwargs.get('eps', 1e-5),
          momentum=kwargs.get('momentum', 0.1),
          affine=kwargs.get('affine', True),
          track_running_stats=kwargs.get('track_running_stats', True))]
    if activation:
      net += [Activation(activation, in_place=True, **kwargs)]
    self.body = nn.Sequential(*net)

  def forward(self, x):
    return self.body(x)

  def initialize_(self, kernel, bias=None):
    """initialize the convolutional weights from external sources

    Args:
        kernel: kernel weight. Shape=[OUT, IN, K, K]
        bias: bias weight. Shape=[OUT]
    """

    dtype = self.body[0].weight.dtype
    device = self.body[0].weight.device
    kernel = torch.tensor(kernel, dtype=dtype, device=device,
                          requires_grad=True)
    assert kernel.shape == self.body[0].weight.shape, "Wrong kernel shape!"
    if bias is not None:
      bias = torch.tensor(bias, dtype=dtype, device=device, requires_grad=True)
      assert bias.shape == self.body[0].bias.shape, "Wrong bias shape!"
    self.body[0].weight.data.copy_(kernel)
    self.body[0].bias.data.copy_(bias)


class RB(nn.Module):
  def __init__(self, in_channels, out_channels=None, kernel_size=3,
               activation=None, use_bias=True, use_bn=False, use_sn=False,
               act_first=None):
    super(RB, self).__init__()
    if out_channels is None:
      out_channels = in_channels
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                      kernel_size // 2, bias=use_bias)
    conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                      kernel_size // 2, bias=use_bias)
    if use_sn:
      conv1 = nn.utils.spectral_norm(conv1)
      conv2 = nn.utils.spectral_norm(conv2)
    net = [conv1, Activation(activation, in_place=True), conv2]
    if use_bn:
      net.insert(1, nn.BatchNorm2d(out_channels))
      if act_first:
        net = [nn.BatchNorm2d(in_channels),
               Activation(activation, in_place=True)] + net
      else:
        net.append(nn.BatchNorm2d(out_channels))
    self.body = nn.Sequential(*net)
    if in_channels != out_channels:
      self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x):
    out = self.body(x)
    if hasattr(self, 'shortcut'):
      sc = self.shortcut(x)
      return out + sc
    return out + x


class Rdb(nn.Module):
  def __init__(self, channels, filters, depth=3, scaling=1.0,
               name='Rdb', **kwargs):
    super(Rdb, self).__init__()
    self.name = name
    self.depth = depth
    self.scaling = scaling
    for i in range(depth):
      conv = EasyConv2d(channels + filters * i, filters, **kwargs)
      setattr(self, f'conv_{i}', conv)
    # no activation after last layer
    try:
      kwargs.pop('activation')
    except KeyError:
      pass
    conv = EasyConv2d(channels + filters * (depth - 1), channels, **kwargs)
    setattr(self, f'conv_{depth - 1}', conv)

  def forward(self, inputs):
    fl = [inputs]
    for i in range(self.depth):
      conv = getattr(self, f'conv_{i}')
      fl.append(conv(torch.cat(fl, dim=1)))
    return fl[-1] * self.scaling + inputs

  def extra_repr(self):
    return f"{self.name}: depth={self.depth}, scaling={self.scaling}"


class Rrdb(nn.Module):
  """
  Residual in Residual Dense Block
  """

  def __init__(self, nc, gc=32, depth=5, scaling=1.0, **kwargs):
    super(Rrdb, self).__init__()
    self.RDB1 = Rdb(nc, gc, depth, scaling, **kwargs)
    self.RDB2 = Rdb(nc, gc, depth, scaling, **kwargs)
    self.RDB3 = Rdb(nc, gc, depth, scaling, **kwargs)
    self.scaling = scaling

  def forward(self, x):
    out = self.RDB1(x)
    out = self.RDB2(out)
    out = self.RDB3(out)
    return out.mul(self.scaling) + x


class Rcab(nn.Module):
  def __init__(self, channels, ratio=16, name='RCAB', **kwargs):
    super(Rcab, self).__init__()
    self.name = name
    self.ratio = ratio
    in_c, out_c = to_list(channels, 2)
    ks = kwargs.get('kernel_size', 3)
    padding = kwargs.get('padding', ks // 2)
    group = kwargs.get('group', 1)
    bias = kwargs.get('bias', True)
    self.c1 = nn.Sequential(
        nn.Conv2d(in_c, out_c, ks, 1, padding, 1, group, bias),
        nn.ReLU(True))
    self.c2 = nn.Conv2d(out_c, out_c, ks, 1, padding, 1, group, bias)
    self.c3 = nn.Sequential(
        nn.Conv2d(out_c, out_c // ratio, 1, groups=group, bias=bias),
        nn.ReLU(True))
    self.c4 = nn.Sequential(
        nn.Conv2d(out_c // ratio, in_c, 1, groups=group, bias=bias),
        nn.Sigmoid())
    self.pooling = nn.AdaptiveAvgPool2d(1)

  def forward(self, inputs):
    x = self.c1(inputs)
    y = self.c2(x)
    x = self.pooling(y)
    x = self.c3(x)
    x = self.c4(x)
    y = x * y
    return inputs + y

  def extra_repr(self):
    return f"{self.name}: ratio={self.ratio}"


class CascadeRdn(nn.Module):
  def __init__(self, channels, filters, depth=3, use_ca=False,
               name='CascadeRdn', **kwargs):
    super(CascadeRdn, self).__init__()
    self.name = name
    self.depth = to_list(depth, 2)
    self.ca = use_ca
    for i in range(self.depth[0]):
      setattr(self, f'conv11_{i}',
              nn.Conv2d(channels + filters * (i + 1), filters, 1))
      setattr(self, f'rdn_{i}', Rdb(channels, filters, self.depth[1], **kwargs))
      if use_ca:
        setattr(self, f'rcab_{i}', Rcab(channels))

  def forward(self, inputs):
    fl = [inputs]
    x = inputs
    for i in range(self.depth[0]):
      rdn = getattr(self, f'rdn_{i}')
      x = rdn(x)
      if self.ca:
        rcab = getattr(self, f'rcab_{i}')
        x = rcab(x)
      fl.append(x)
      c11 = getattr(self, f'conv11_{i}')
      x = c11(torch.cat(fl, dim=1))

    return x

  def extra_repr(self):
    return f"{self.name}: depth={self.depth}, ca={self.ca}"


class CBAM(nn.Module):
  """Convolutional Block Attention Module (ECCV 18)
  - CA: channel attention module
  - SA: spatial attention module

  Args:
    channels: input channel of tensors
    channel_reduction: reduction ratio in `CA`
    spatial_first: put SA ahead of CA (default: CA->SA)
  """

  class CA(nn.Module):
    def __init__(self, channels, ratio=16):
      super(CBAM.CA, self).__init__()
      self.max_pool = nn.AdaptiveMaxPool2d(1)
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.mlp = nn.Sequential(
          nn.Conv2d(channels, channels // ratio, 1),
          nn.ReLU(),
          nn.Conv2d(channels // ratio, channels, 1))

    def forward(self, x):
      maxpool = self.max_pool(x)
      avgpool = self.avg_pool(x)
      att = F.sigmoid(self.mlp(maxpool) + self.mlp(avgpool))
      return att * x

  class SA(nn.Module):
    def __init__(self, kernel_size=7):
      super(CBAM.SA, self).__init__()
      self.conv = nn.Conv2d(2, 1, kernel_size, 1, kernel_size // 2)

    def forward(self, x):
      max_c_pool = x.max(dim=1, keepdim=True)
      avg_c_pool = x.mean(dim=1, keepdim=True)
      y = torch.cat([max_c_pool, avg_c_pool], dim=1)
      att = F.sigmoid(self.conv(y))
      return att * x

  def __init__(self, channels, channel_reduction=16, spatial_first=None):
    super(CBAM, self).__init__()
    self.channel_attention = CBAM.CA(channels, ratio=channel_reduction)
    self.spatial_attention = CBAM.SA(7)
    self.spatial_first = spatial_first

  def forward(self, inputs):
    if self.spatial_first:
      x = self.spatial_attention(inputs)
      return self.channel_attention(x)
    else:
      x = self.channel_attention(inputs)
      return self.spatial_attention(x)


class Conv2dLSTMCell(nn.Module):
  """ConvLSTM cell.
  Copied from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81
  Special thanks to @Kaixhin
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True):
    super(Conv2dLSTMCell, self).__init__()
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.padding_h = tuple(
        k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
    self.dilation = dilation
    self.groups = groups
    self.weight_ih = Parameter(
        torch.Tensor(4 * out_channels, in_channels // groups, *kernel_size))
    self.weight_hh = Parameter(
        torch.Tensor(4 * out_channels, out_channels // groups, *kernel_size))
    self.weight_ch = Parameter(
        torch.Tensor(3 * out_channels, out_channels // groups, *kernel_size))
    if bias:
      self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
      self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
      self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
    else:
      self.register_parameter('bias_ih', None)
      self.register_parameter('bias_hh', None)
      self.register_parameter('bias_ch', None)
    self.register_buffer('wc_blank', torch.zeros(out_channels))
    self.reset_parameters()

  def reset_parameters(self):
    n = 4 * self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight_ih.data.uniform_(-stdv, stdv)
    self.weight_hh.data.uniform_(-stdv, stdv)
    self.weight_ch.data.uniform_(-stdv, stdv)
    if self.bias_ih is not None:
      self.bias_ih.data.uniform_(-stdv, stdv)
      self.bias_hh.data.uniform_(-stdv, stdv)
      self.bias_ch.data.uniform_(-stdv, stdv)

  def forward(self, input, hx):
    h_0, c_0 = hx

    wx = F.conv2d(input, self.weight_ih, self.bias_ih, self.stride,
                  self.padding, self.dilation, self.groups)
    wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                  self.padding_h, self.dilation, self.groups)
    # Cell uses a Hadamard product instead of a convolution?
    wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                  self.padding_h, self.dilation, self.groups)
    v = Variable(self.wc_blank).reshape((1, -1, 1, 1))
    wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels],
                                v.expand(wc.size(0), wc.size(1) // 3,
                                         wc.size(2), wc.size(3)),
                                wc[:, 2 * self.out_channels:]), 1)

    i = torch.sigmoid(wxhc[:, :self.out_channels])
    f = torch.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
    g = torch.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
    o = torch.sigmoid(wxhc[:, 3 * self.out_channels:])

    c_1 = f * c_0 + i * g
    h_1 = o * torch.tanh(c_1)
    return h_1, (h_1, c_1)
