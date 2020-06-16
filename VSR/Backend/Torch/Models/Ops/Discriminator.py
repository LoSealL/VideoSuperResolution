#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

from torch import nn

from .Blocks import Activation, EasyConv2d, RB


def _pull_conv_args(**kwargs):
  f = kwargs.get('filters', 64)
  ks = kwargs.get('kernel_size', 3)
  activation = kwargs.get('activation', 'leaky')
  bias = kwargs.get('bias', True)
  norm = kwargs.get('norm', '')
  bn = norm.lower() in ('bn', 'batch')
  sn = norm.lower() in ('sn', 'spectral')
  return f, ks, activation, bias, bn, sn


class DCGAN(nn.Module):
  """DCGAN-like discriminator:
    stack of conv2d layers, stride down to 4x4

  Args:
    channel: input tensor channel
    num_layers: number of total cnn layers
    norm: could be "None", "SN/Spectral" or "BN/Batch"
    leaky: leaky slope
    favor: some pre-defined topology:
      'A': s1 s2 s1 s2 ...
      'B': s1 s2 s2 s2 ...
    kwargs: additional options to common CNN

  Note: Since the input before FC layer is B*N*4*4, the input shape can be
    derived as 4 * (2 ** n_strided), where $n_{strided}=num_layers / 2$ in
    favor 'A' and $n_{strided}=num_layers - 1$ in favor 'B'.
  """

  def __init__(self, channel, num_layers, norm=None, favor='A', **kwargs):
    super(DCGAN, self).__init__()
    f, ks, act, bias, bn, sn = _pull_conv_args(norm=norm, **kwargs)
    net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn,
                      use_bias=bias)]
    self.n_strided = 0
    counter = 1
    assert favor in ('A', 'B', 'C'), "favor must be A | B | C"
    while True:
      f *= 2
      net.append(EasyConv2d(
        f // 2, f, ks + 1, 2, activation=act, use_bias=bias, use_bn=bn,
        use_sn=sn))
      self.n_strided += 1
      counter += 1
      if counter >= num_layers:
        break
      if favor in ('A', 'C'):
        net.append(EasyConv2d(
          f, f, ks, 1, activation=act, use_bias=bias, use_bn=bn,
          use_sn=sn))
        counter += 1
        if counter >= num_layers:
          break
    if favor == 'C':
      self.body = nn.Sequential(*net, nn.AdaptiveAvgPool2d(1))
      linear = [nn.Linear(f, 100, bias),
                Activation(act, in_place=True),
                nn.Linear(100, 1, bias)]
    else:
      self.body = nn.Sequential(*net)
      linear = [nn.Linear(f * 4 * 4, 100, bias),
                Activation(act, in_place=True),
                nn.Linear(100, 1, bias)]
    if sn:
      linear[0] = nn.utils.spectral_norm(linear[0])
      linear[2] = nn.utils.spectral_norm(linear[2])
    self.linear = nn.Sequential(*linear)

  def forward(self, x):
    # assert x.size(2) == x.size(3) == 4 * 2 ** self.n_strided
    y = self.body(x).flatten(1)
    return self.linear(y)


class Residual(nn.Module):
  """Resnet-like discriminator.
    Stack of residual block, avg_pooling down to 4x4.

  Args:
    channel: input tensor channel
    num_residual: number of total cnn layers
    norm: could be "None", "SN/Spectral" or "BN/Batch"
    leaky: leaky slope
    favor: some pre-defined topology:
      'A': norm before 1st conv in residual
      'B': norm after 2nd conv in residual
    kwargs: additional options to common CNN

  Note: there is always activation and norm after 1st conv; if channel mis-
    matches, a 1x1 conv is used for shortcut
  """

  def __init__(self, channel, num_residual, norm=None, favor='A', **kwargs):
    super(Residual, self).__init__()
    f, ks, act, bias, bn, sn = _pull_conv_args(norm=norm, **kwargs)
    net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn,
                      use_bias=bias)]
    for i in range(num_residual):
      net.append(RB(f, ks, act, bias, bn, sn, favor == 'A'))
      net.append(nn.AvgPool2d(2))
    net.append(Activation(act, in_place=True))
    self.body = nn.Sequential(*net)
    linear = [nn.Linear(f * 4 * 4, 100, bias),
              Activation(act, in_place=True),
              nn.Linear(100, 1, bias)]
    if sn:
      linear[0] = nn.utils.spectral_norm(linear[0])
      linear[2] = nn.utils.spectral_norm(linear[2])
    self.linear = nn.Sequential(*linear)
    self.n_strided = num_residual

  def forward(self, x):
    assert x.size(2) == x.size(3) == 4 * 2 ** self.n_strided
    y = self.body(x).flatten(1)
    return self.linear(y)
