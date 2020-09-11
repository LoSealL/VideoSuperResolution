"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-11

EDRN is implemented by IVIP-Lab.
@yyknight https://github.com/yyknight/NTIRE2019_EDRN
"""
import logging

import torch
import torch.nn as nn

_logger = logging.getLogger("VSR.EDRN")


class CALayer(nn.Module):
    def __init__(self, channel, wn, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        del std


class Residual_Block(nn.Module):
    def __init__(self, inChannels, growRate0, wn, kSize=3, stride=1):
        super(Residual_Block, self).__init__()
        Cin = inChannels
        G0 = growRate0
        self.conv = nn.Sequential(*[
            wn(nn.Conv2d(Cin, G0, kSize, padding=(kSize - 1) // 2, stride=stride)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=stride)),
            CALayer(Cin, wn, 16)
        ])

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out


class RG(nn.Module):
    def __init__(self, growRate0, nConvLayers, wn, kSize=3):
        super(RG, self).__init__()
        G0 = growRate0
        C = nConvLayers
        convs_residual = []
        for c in range(C):
            convs_residual.append(Residual_Block(G0, G0, wn))
        self.convs_residual = nn.Sequential(*convs_residual)
        self.last_conv = wn(
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1))

    def forward(self, x):
        x = self.last_conv(self.convs_residual(x)) + x
        return x


class EDRN(nn.Module):
    def __init__(self, args):
        _logger.info("LICENSE: EDRN is implemented by IVIP-Lab. "
                     "@yyknight https://github.com/yyknight/NTIRE2019_EDRN")
        super(EDRN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.EDRNkSize
        rgb_mean = (0.4313, 0.4162, 0.3861)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        def wn(x): return torch.nn.utils.weight_norm(x)
        # number of RG blocks, conv layers, out channels
        self.D, C, G = {
            'B': (4, 10, 16),
        }[args.EDRNconfig]

        self.SFENet1 = wn(
            nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1))
        self.encoder1 = nn.Sequential(*[
            wn(nn.Conv2d(G0, 2 * G0, kSize, padding=(kSize - 1) // 2, stride=2)),
            nn.BatchNorm2d(2 * G0),
            nn.ReLU(inplace=True)])
        self.encoder2 = nn.Sequential(*[
            wn(nn.Conv2d(2 * G0, 4 * G0, kSize, padding=(kSize - 1) // 2, stride=2)),
            nn.BatchNorm2d(4 * G0),
            nn.ReLU(inplace=True)
        ])
        self.decoder1 = nn.Sequential(*[
            wn(nn.ConvTranspose2d(4 * G0, 2 * G0, 3, padding=1, output_padding=1,
                                  stride=2)),
            nn.BatchNorm2d(2 * G0),
            nn.ReLU(inplace=True)])
        self.decoder2 = nn.Sequential(*[
            wn(nn.ConvTranspose2d(2 * G0, G0, 3, padding=1, output_padding=1,
                                  stride=2)),
            nn.BatchNorm2d(G0),
            nn.ReLU()
        ])
        RGs0 = [
            RG(growRate0=4 * G0, nConvLayers=C, wn=wn)
            for _ in range(self.D)]
        RGs0.append(
            wn(nn.Conv2d(4 * G0, 4 * G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        RGs1 = [
            RG(growRate0=2 * G0, nConvLayers=C, wn=wn)
            for _ in range(self.D // 2)]
        RGs1.append(
            wn(nn.Conv2d(2 * G0, 2 * G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        RGs2 = [
            RG(growRate0=G0, nConvLayers=C, wn=wn)
            for _ in range(self.D // 4)]
        RGs2.append(
            wn(nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        self.RGs0 = nn.Sequential(*RGs0)
        self.RGs1 = nn.Sequential(*RGs1)
        self.RGs2 = nn.Sequential(*RGs2)
        self.restoration = wn(
            nn.Conv2d(G0, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1))
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        f__2 = self.encoder1(f__1)
        f__3 = self.encoder2(f__2)
        x = f__3
        x = self.decoder1(self.RGs0(x) + f__3)
        x = self.decoder2(self.RGs1(x) + f__2)
        x = self.RGs2(x) + f__1
        x = self.restoration(x)
        x = self.add_mean(x)

        return x
