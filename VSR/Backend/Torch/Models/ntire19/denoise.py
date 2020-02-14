#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/20 上午11:47

import logging

import torch
import torch.nn as nn

from ..Arch import Activation, Rdb, SpaceToDepth, CBAM

_logger = logging.getLogger("VSR.NTIRE2019.Denoise")


class EraserTeam:
  class URB(nn.Module):
    def __init__(self, filters, depth, act=None):
      super(EraserTeam.URB, self).__init__()
      convs = []
      for i in range(depth):
        convs.append(nn.Conv2d(filters, filters, 3, 1, 1))
        if act:
          convs.append(Activation(act))
      self.body = nn.Sequential(*convs)

    def forward(self, x):
      return x + self.body(x)

  class UModule(nn.Module):
    def __init__(self, filters):
      super(EraserTeam.UModule, self).__init__()
      self.head = EraserTeam.URB(filters, 2, 'PReLU')
      self.down = nn.Sequential(
        nn.Conv2d(filters, filters * 2, 3, 2, 1),
        EraserTeam.URB(filters * 2, 1, 'PReLU'))
      self.downup = nn.Sequential(
        nn.Conv2d(filters * 2, filters * 4, 3, 2, 1),
        EraserTeam.URB(filters * 4, 1, 'PReLU'),
        nn.Conv2d(filters * 4, filters * 8, 1), nn.PixelShuffle(2))
      self.up = nn.Sequential(
        nn.Conv2d(filters * 4, filters * 2, 1),
        EraserTeam.URB(filters * 2, 1, 'PReLU'),
        nn.Conv2d(filters * 2, filters * 4, 1), nn.PixelShuffle(2))
      self.tail = nn.Sequential(
        nn.Conv2d(filters * 2, filters, 1),
        EraserTeam.URB(filters, 2, 'PReLU'),
        nn.Conv2d(filters, filters, 3, 1, 1), nn.PReLU())

    def forward(self, inputs):
      c0 = inputs
      c1 = self.head(inputs)
      c2 = self.down(c1)
      c3 = self.downup(c2)
      c4 = self.up(torch.cat([c3, c2], dim=1))
      c5 = self.tail(torch.cat([c4, c1], dim=1))
      return c5 + c0

  class DIDN(nn.Module):
    def __init__(self, channels=3, filters=32, n_modules=8):
      _logger.info("DIDN was introduced by Songhyun Yu @Hanyang University, "
                   "Implemented by @LoSealL")
      super(EraserTeam.DIDN, self).__init__()
      self.head = nn.Sequential(
        nn.Conv2d(channels, filters, 3, 1, 1), nn.PReLU(),
        nn.Conv2d(filters, filters * 2, 3, 2, 1))
      self.tail = nn.Sequential(
        nn.Conv2d(filters * n_modules, filters * 4, 1),
        EraserTeam.URB(filters * 4, 1, 'PReLU'),
        nn.PixelShuffle(2),
        nn.Conv2d(filters, channels, 3, 1, 1), nn.PReLU())
      self.recon = nn.Sequential(
        nn.Conv2d(filters * 2, filters, 3, 1, 1), nn.PReLU())
      for i in range(n_modules):
        setattr(self, f'umod_{i:02d}', EraserTeam.UModule(filters * 2))
      self.nu = n_modules

    def forward(self, inputs):
      c0 = inputs
      x = self.head(c0)
      u_out = []
      for i in range(self.nu):
        x = getattr(self, f'umod_{i:02d}')(x)
        u_out.append(x)
      u_out = [self.recon(t) for t in u_out]
      x = self.tail(torch.cat(u_out, dim=1))
      return c0 + x

  class DCR(nn.Module):
    def __init__(self, filters):
      super(EraserTeam.DCR, self).__init__()
      self.conv1 = nn.Sequential(
        nn.Conv2d(filters, filters // 2, 3, 1, 1), nn.PReLU())
      self.conv2 = nn.Sequential(
        nn.Conv2d(filters * 3 // 2, filters // 2, 3, 1, 1), nn.PReLU())
      self.conv3 = nn.Sequential(
        nn.Conv2d(filters * 2, filters, 3, 1, 1), nn.PReLU())

    def forward(self, x):
      c0 = x
      c1 = self.conv1(c0)
      c2 = self.conv2(torch.cat([c0, c1], dim=1))
      c3 = self.conv3(torch.cat([c0, c1, c2], dim=1))
      return c3 + c0

  class DHDN(nn.Module):
    def __init__(self, channels=3, filters=128):
      _logger.info("DHDN was introduced by Songhyun Yu @Hanyang University, "
                   "Implemented by @LoSealL")
      super(EraserTeam.DHDN, self).__init__()
      self.entry = nn.Sequential(
        nn.Conv2d(channels, filters, 1), nn.PReLU(),
        EraserTeam.DCR(filters), EraserTeam.DCR(filters))
      self.exit = nn.Sequential(
        EraserTeam.DCR(filters * 2), EraserTeam.DCR(filters * 2),
        nn.Conv2d(filters * 2, channels, 1), nn.PReLU())
      self.down1 = nn.Sequential(
        nn.Conv2d(filters, filters * 2, 3, 2, 1),
        EraserTeam.DCR(filters * 2), EraserTeam.DCR(filters * 2))
      self.down2 = nn.Sequential(
        nn.Conv2d(filters * 2, filters * 4, 3, 2, 1),
        EraserTeam.DCR(filters * 4), EraserTeam.DCR(filters * 4))
      self.downup = nn.Sequential(
        nn.Conv2d(filters * 4, filters * 8, 3, 2, 1),
        EraserTeam.DCR(filters * 8), EraserTeam.DCR(filters * 8),
        nn.ConvTranspose2d(filters * 8, filters * 4, 3, 2, 1, 1))
      self.up1 = nn.Sequential(
        EraserTeam.DCR(filters * 8), EraserTeam.DCR(filters * 8),
        nn.ConvTranspose2d(filters * 8, filters * 2, 3, 2, 1, 1))
      self.up2 = nn.Sequential(
        EraserTeam.DCR(filters * 4), EraserTeam.DCR(filters * 4),
        nn.ConvTranspose2d(filters * 4, filters, 3, 2, 1, 1))

    def forward(self, inputs):
      c0 = inputs
      c1 = self.entry(c0)
      c2 = self.down1(c1)
      c3 = self.down2(c2)
      c4 = self.downup(c3)
      c5 = self.up1(torch.cat([c4, c3], dim=1))
      c6 = self.up2(torch.cat([c5, c2], dim=1))
      c7 = self.exit(torch.cat([c6, c1], dim=1))
      return c7 + c0


class DGUTeam:
  class GRDB(nn.Module):
    def __init__(self, n_rdb, filters):
      super(DGUTeam.GRDB, self).__init__()
      for i in range(n_rdb):
        setattr(self, f'rdb_{i:02d}', Rdb(filters))
      self.n_rdb = n_rdb
      self.conv = nn.Conv2d(filters * n_rdb, filters, 1)

    def forward(self, x):
      rdb_o = [x]
      for i in range(self.n_rdb):
        rdb_o.append(getattr(self, f'rdb_{i:02d}')(rdb_o[-1]))
      return x + self.conv(torch.cat(rdb_o[1:], dim=1))

  class GRDN(nn.Module):
    def __init__(self, channels, filters=64, n_grdb=10, n_rdb=4):
      _logger.info("GRDN was introduced by Seung-Won Jung @Dongguk University, "
                   "Implemented by @LoSealL")
      super(DGUTeam.GRDN, self).__init__()
      nets = [nn.Conv2d(channels, filters, 3, 1, 1)]
      for _ in range(n_grdb):
        nets.append(DGUTeam.GRDB(n_rdb, filters))
      nets.append(nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1))
      self.head = nn.Sequential(*nets)
      self.cbam = CBAM(filters)
      self.tail = nn.Conv2d(filters, channels, 3, 1, 1)

    def forward(self, inputs):
      c0 = inputs
      x = self.head(c0)
      x = self.cbam(x)
      x = self.tail(x)
      return x + c0


class HITVPCTeam:
  class RB(nn.Module):
    def __init__(self, filters):
      super(HITVPCTeam.RB, self).__init__()
      self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
      self.act = nn.ReLU(True)
      self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)

    def forward(self, x):
      c0 = x
      x = self.conv1(x)
      x = self.act(x)
      x = self.conv2(x)
      return x + c0

  class NRB(nn.Module):
    def __init__(self, n, f):
      super(HITVPCTeam.NRB, self).__init__()
      nets = []
      for i in range(n):
        nets.append(HITVPCTeam.RB(f))
      self.body = nn.Sequential(*nets)

    def forward(self, x):
      return self.body(x)

  class ResUNet(nn.Module):
    def __init__(self, channels, filters=128, n_rb=10):
      _logger.info("ResUNet was introduced by Kai Zhang @HIT, "
                   "Implemented by @LoSealL")
      super(HITVPCTeam.ResUNet, self).__init__()
      self.head = nn.Sequential(
        SpaceToDepth(2),
        nn.Conv2d(channels * 4, filters, 3, 1, 1),
        nn.ReLU(True))
      self.down1 = nn.Sequential(
        HITVPCTeam.NRB(n_rb, filters),
        nn.Conv2d(filters, filters, 3, 2, 1), nn.ReLU(True))
      self.down2 = nn.Sequential(
        HITVPCTeam.NRB(n_rb, filters),
        nn.Conv2d(filters, filters, 3, 2, 1), nn.ReLU(True))
      self.down3 = nn.Sequential(
        HITVPCTeam.NRB(n_rb, filters),
        nn.Conv2d(filters, filters, 3, 2, 1), nn.ReLU(True))
      self.middle = HITVPCTeam.NRB(n_rb, filters)
      self.up1 = nn.Sequential(
        nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.ReLU(True),
        HITVPCTeam.NRB(n_rb, filters))
      self.up2 = nn.Sequential(
        nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.ReLU(True),
        HITVPCTeam.NRB(n_rb, filters))
      self.up3 = nn.Sequential(
        nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.ReLU(True),
        HITVPCTeam.NRB(n_rb, filters))
      self.tail = nn.Sequential(
        nn.Conv2d(filters, channels * 4, 3, 1, 1),
        nn.PixelShuffle(2))

    def forward(self, inputs):
      c0 = inputs  # 512
      c1 = self.head(c0)  # 256
      c2 = self.down1(c1)  # 128
      c3 = self.down2(c2)  # 64
      c4 = self.down3(c3)  # 32
      m = self.middle(c4) + c4
      c5 = self.up1(m) + c3
      c6 = self.up2(c5) + c2
      c7 = self.up3(c6) + c1
      return self.tail(c7) + c0
