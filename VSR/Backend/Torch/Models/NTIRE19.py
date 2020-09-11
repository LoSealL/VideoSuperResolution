"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-6

EDRN, FRN, RAN2
"""
from VSR.Util.Config import Config

from .Contrib.ntire19 import denoise, edrn, frn, ran2
from .Optim.SISR import L1Optimizer


class EDRN(L1Optimizer):
    """EDRN is one candidate of NTIRE19 RSR"""

    def __init__(self, scale, channel, **kwargs):
        args = Config(kwargs)
        args.scale = [scale]
        args.n_colors = channel
        self.rgb_range = args.rgb_range
        self.edrn = edrn.EDRN(args)
        super(EDRN, self).__init__(channel=channel, scale=scale, **kwargs)

    def fn(self, x):
        return self.edrn(x * self.rgb_range) / self.rgb_range


class FRN(L1Optimizer):
    def __init__(self, scale, channel, **kwargs):
        args = Config(kwargs)
        args.scale = [scale]
        args.n_colors = channel
        self.rgb_range = args.rgb_range
        self.frn = frn.FRN_UPDOWN(args)
        super(FRN, self).__init__(channel=channel, scale=scale, **kwargs)

    def fn(self, x):
        return self.frn(x * self.rgb_range) / self.rgb_range


class RAN(L1Optimizer):
    def __init__(self, scale, channel, **kwargs):
        args = Config(kwargs)
        args.scale = [scale]
        args.n_colors = channel
        self.rgb_range = args.rgb_range
        self.ran = ran2.RAN(args)
        super(RAN, self).__init__(channel=channel, scale=scale, **kwargs)

    def fn(self, x):
        return self.ran(x * self.rgb_range) / self.rgb_range


class DIDN(L1Optimizer):
    def __init__(self, channel, filters, umodule, **kwargs):
        self.didn = denoise.EraserTeam.DIDN(channel, filters, umodule)
        super(DIDN, self).__init__(channel=channel, **kwargs)

    def fn(self, x):
        return self.didn(x)


class DHDN(L1Optimizer):
    def __init__(self, channel, filters, **kwargs):
        self.dhdn = denoise.EraserTeam.DHDN(channel, filters)
        super(DHDN, self).__init__(channel=channel, **kwargs)

    def fn(self, x):
        return self.dhdn(x)


class GRDN(L1Optimizer):
    def __init__(self, channel, filters, grdb, rdb, **kwargs):
        self.grdn = denoise.DGUTeam.GRDN(channel, filters, grdb, rdb)
        super(GRDN, self).__init__(channel=channel, **kwargs)

    def fn(self, x):
        return self.grdn(x)


class ResUNet(L1Optimizer):
    def __init__(self, channel, filters, rb, **kwargs):
        self.resunet = denoise.HITVPCTeam.ResUNet(channel, filters, rb)
        super(ResUNet, self).__init__(channel=channel, **kwargs)

    def fn(self, x):
        return self.resunet(x)
