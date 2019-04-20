#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 4 - 16

import torch.nn as nn

from ..edsr import common


## Channel Attention (CA) Layer
class CALayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(CALayer, self).__init__()
    # global average pooling: feature --> point
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    # feature channel downscale and upscale --> channel weight
    self.conv_du = nn.Sequential(
      nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
    )

    self.sigmoid = nn.Sigmoid()
    self.ch = channel

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.conv_du(y)
    y = self.sigmoid(y)

    return x * y
  ## Residual Channel Attention Block (RCAB)


class RCAB(nn.Module):
  def __init__(
      self, conv, n_feat, kernel_size, reduction,
      bias=True, bn=False, act=nn.LeakyReLU(0.2, True), res_scale=1, px=1):

    super(RCAB, self).__init__()
    modules_body = []
    # modules_body.append(common.invPixelShuffle(2))
    modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
    modules_body.append(act)
    modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

    if px != 1:
      modules_body.append(common.invPixelShuffle(px))
    modules_body.append(CALayer(n_feat * px ** 2, reduction))
    if px != 1:
      modules_body.append(nn.PixelShuffle(px))
    self.body = nn.Sequential(*modules_body)
    self.res_scale = res_scale

  def forward(self, x):
    res = self.body(x)
    # res = self.body(x).mul(self.res_scale)
    res += x
    return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
  def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
               n_resblocks, n_resgroups, px):
    super(ResidualGroup, self).__init__()
    modules_body = []
    if len(n_resgroups) == 0:
      modules_body = [
        RCAB(
          conv, n_feat, kernel_size, reduction, bias=True, bn=False,
          act=nn.ReLU(True), res_scale=1, px=px) \
        for _ in range(n_resblocks)]
    else:
      modules_body = [
        ResidualGroup(conv, n_feat, kernel_size, reduction, act, res_scale,
                      n_resblocks, n_resgroups[1:], px=px) \
        for _ in range(n_resgroups[0])]

    modules_body.append(conv(n_feat, n_feat, kernel_size))
    self.body = nn.Sequential(*modules_body)

  def forward(self, x):
    res = self.body(x)
    res += x
    return res


## Residual Channel Attention Network (RCAN)
class FRN_UPDOWN(nn.Module):
  def __init__(self, args, conv=common.default_conv):
    super(FRN_UPDOWN, self).__init__()

    n_resgroups = args.n_resgroups
    # n_resgroups_ = args.n_resgroups2
    # n_resgroups__ = args.n_resgroups3

    n_resblocks = args.n_resblocks
    n_feats = args.n_feats
    kernel_size = 3
    reduction = args.reduction
    scale = args.scale[0]
    act = nn.ReLU(True)

    # RGB mean for DIV2K
    rgb_mean = (0.4488, 0.4371, 0.4040)
    rgb_std = (1.0, 1.0, 1.0)
    self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

    # define head module
    modules_head = [conv(args.n_colors, n_feats, kernel_size)]

    # define body module
    modules_body = [
      ResidualGroup(
        conv, n_feats, kernel_size, reduction, act=act,
        res_scale=args.res_scale, n_resblocks=n_resblocks,
        n_resgroups=n_resgroups[1:], px=args.px) \
      for _ in range(n_resgroups[0])]

    modules_body.append(conv(n_feats, n_feats, kernel_size))
    # define tail module
    modules_tail = [
      common.Upsampler(conv, scale, n_feats, act=False),
      conv(n_feats, args.n_colors, kernel_size)]

    self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
    m_down = [
      common.invUpsampler(conv, scale, n_feats, act=False),
    ]

    self.down = nn.Sequential(*m_down)

    # self.head = nn.Sequential(*modules_head)
    self.new_head = nn.Sequential(*modules_head)
    self.body = nn.Sequential(*modules_body)
    # self.tail = nn.Sequential(*modules_tail)
    self.new_tail = nn.Sequential(*modules_tail)

    for m in self.body.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)

  def forward(self, x):
    x = self.sub_mean(x)
    x = self.new_head(x)
    x = self.down(x)

    res = self.body(x)
    res += x

    x = self.new_tail(res)
    x = self.add_mean(x)

    return x

  def forward_(self, x):

    x = self.sub_mean(x)
    x = self.new_head(x)
    x = self.down(x)

    x = self.new_tail(x)
    x = self.add_mean(x)

    return x

  def load_state_dict(self, state_dict, strict=False):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name in own_state:
        if isinstance(param, nn.Parameter):
          param = param.data
        try:
          own_state[name].copy_(param)
        except Exception:
          if name.find('tail') >= 0:
            print('Replace pre-trained upsampler to new one...')
          else:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(name, own_state[name].size(),
                                       param.size()))
      elif strict:
        if name.find('tail') == -1:
          raise KeyError('unexpected key "{}" in state_dict'
                         .format(name))

    if strict:
      missing = set(own_state.keys()) - set(state_dict.keys())
      if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))
