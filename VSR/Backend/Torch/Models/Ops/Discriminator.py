#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

from torch import nn

from .Blocks import Activation, EasyConv2d, RB


def _pull_conv_args(**kwargs):
  def _get_and_pop(d: dict, key, default=None):
    if key in d:
      return d.pop(key)
    return d.get(key, default)

  f = _get_and_pop(kwargs, 'filters', 64)
  ks = _get_and_pop(kwargs, 'kernel_size', 3)
  activation = _get_and_pop(kwargs, 'activation', 'leaky')
  bias = _get_and_pop(kwargs, 'bias', True)
  norm = _get_and_pop(kwargs, 'norm', '')
  bn = norm.lower() in ('bn', 'batch')
  sn = norm.lower() in ('sn', 'spectral')
  return f, ks, activation, bias, bn, sn, kwargs


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

  def __init__(self, channel, num_layers, scale=4, norm=None, favor='A',
               **kwargs):
    super(DCGAN, self).__init__()
    f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
    net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn,
                      use_bias=bias, negative_slope=0.2)]
    self.n_strided = 0
    counter = 1
    assert favor in ('A', 'B', 'C'), "favor must be A | B | C"
    while True:
      f *= 2
      net.append(EasyConv2d(
          f // 2, f, ks + 1, 2, activation=act, use_bias=bias, use_bn=bn,
          use_sn=sn, **unparsed))
      self.n_strided += 1
      counter += 1
      if counter >= num_layers:
        break
      if favor in ('A', 'C'):
        net.append(EasyConv2d(
            f, f, ks, 1, activation=act, use_bias=bias, use_bn=bn,
            use_sn=sn, **unparsed))
        counter += 1
        if counter >= num_layers:
          break
    if favor == 'C':
      self.body = nn.Sequential(*net, nn.AdaptiveAvgPool2d(1))
      linear = [nn.Linear(f, 100, bias),
                Activation(act, in_place=True, **unparsed),
                nn.Linear(100, 1, bias)]
    else:
      self.body = nn.Sequential(*net)
      linear = [nn.Linear(f * scale * scale, 100, bias),
                Activation(act, in_place=True, **unparsed),
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
    f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
    net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn,
                      use_bias=bias, **unparsed)]
    for i in range(num_residual):
      net.append(RB(f, ks, act, bias, bn, sn, favor == 'A'))
      net.append(nn.AvgPool2d(2))
    net.append(Activation(act, in_place=True, **unparsed))
    self.body = nn.Sequential(*net)
    linear = [nn.Linear(f * 4 * 4, 100, bias),
              Activation(act, in_place=True, **unparsed),
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


class PatchGAN(nn.Module):
  """Defines a PatchGAN discriminator
  Args:
      channel: the number of channels in input images
      num_layers: number of total cnn layers
      norm: could be "None", "SN/Spectral" or "BN/Batch"
  """

  def __init__(self, channel, num_layers=3, norm=None, **kwargs):
    super(PatchGAN, self).__init__()
    f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
    sequence = [
      EasyConv2d(channel, f, ks + 1, 2, activation=act, use_bn=bn, use_sn=sn,
                 use_bias=bias, **unparsed)]
    in_c = f
    out_c = f * 2
    for n in range(1, num_layers):
      sequence.append(
          EasyConv2d(in_c, out_c, ks + 1, 2, activation=act, use_bn=bn,
                     use_sn=sn, use_bias=bias, **unparsed))
      in_c = out_c
      out_c *= 2
    sequence += [
      EasyConv2d(in_c, out_c, ks, activation=act, use_bn=bn, use_sn=sn,
                 use_bias=bias, **unparsed),
      EasyConv2d(out_c, 1, 1)
    ]
    self.body = nn.Sequential(*sequence)

  def forward(self, x):
    return self.body(x)
