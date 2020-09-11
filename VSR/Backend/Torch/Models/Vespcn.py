"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-3

VESPCN
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Framework.Summary import get_writer
from ..Util import pad_if_divide, psnr, total_variance
from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d
from .Ops.Motion import STN, CoarseFineFlownet

_logger = logging.getLogger("VSR.VESPCN")
_logger.info("LICENSE: VESPCN is proposed at CVPR2017 by Twitter. "
             "Implemented by myself @LoSealL.")


class ReluRB(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ReluRB, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, outchannels, 3, 1, 1)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs


class MotionCompensation(nn.Module):
    def __init__(self, channel, gain=32):
        super(MotionCompensation, self).__init__()
        self.gain = gain
        self.flownet = CoarseFineFlownet(channel)
        self.warp_f = STN(padding_mode='border')

    def forward(self, target, ref):
        flow = self.flownet(target, ref, self.gain)
        warping = self.warp_f(ref, flow[:, 0], flow[:, 1])
        return warping, flow


class SRNet(nn.Module):
    def __init__(self, scale, channel, depth):
        super(SRNet, self).__init__()
        self.entry = EasyConv2d(channel * depth, 64, 3)
        self.exit = EasyConv2d(64, channel, 3)
        self.body = nn.Sequential(
            ReluRB(64, 64),
            ReluRB(64, 64),
            ReluRB(64, 64),
            nn.ReLU(True))
        self.conv = EasyConv2d(64, 64 * scale ** 2, 3)
        self.up = nn.PixelShuffle(scale)

    def forward(self, inputs):
        x = self.entry(inputs)
        y = self.body(x) + x
        y = self.conv(y)
        y = self.up(y)
        y = self.exit(y)
        return y


class Vespcn(nn.Module):
    def __init__(self, scale, channel, depth):
        super(Vespcn, self).__init__()
        self.sr = SRNet(scale, channel, depth)
        self.mc = MotionCompensation(channel)
        self.depth = depth

    def forward(self, *inputs):
        center = self.depth // 2
        target = inputs[center]
        refs = inputs[:center] + inputs[center + 1:]
        warps = []
        flows = []
        for r in refs:
            warp, flow = self.mc(target, r)
            warps.append(warp)
            flows.append(flow)
        warps.append(target)
        x = torch.cat(warps, 1)
        sr = self.sr(x)
        return sr, warps[:-1], flows


class VESPCN(SuperResolution):
    def __init__(self, scale, channel, depth=3, **kwargs):
        super(VESPCN, self).__init__(scale, channel, **kwargs)
        self.vespcn = Vespcn(scale, channel, depth)
        self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
        self.depth = depth

    def train(self, inputs, labels, learning_rate=None):
        frames = torch.split(inputs[0], 1, dim=1)
        frames = [f.squeeze(1) for f in frames]
        sr, warps, flows = self.vespcn(*frames)
        targets = torch.split(labels[0], 1, dim=1)
        targets = [t.squeeze(1) for t in targets]
        target = targets[self.depth // 2]
        ref = frames[self.depth // 2]

        loss_content = F.mse_loss(sr, target)
        loss_flow = torch.sum(torch.stack([F.mse_loss(ref, w) for w in warps]))
        loss_tv = torch.sum(torch.stack(
            [total_variance(f) for f in flows]))

        loss = loss_content + loss_flow + 0.01 * loss_tv
        if learning_rate:
            for param_group in self.opt.param_groups:
                param_group["lr"] = learning_rate
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            'image': loss_content.detach().cpu().numpy(),
            'flow': loss_flow.detach().cpu().numpy(),
            'tv': loss_tv.detach().cpu().numpy(),
        }

    def eval(self, inputs, labels=None, **kwargs):
        metrics = {}
        frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
        _frames = [pad_if_divide(x, 4, 'reflect') for x in frames]
        a = (_frames[0].size(2) - frames[0].size(2)) * self.scale
        b = (_frames[0].size(3) - frames[0].size(3)) * self.scale
        slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
        slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
        sr, warps, flows = self.vespcn(*_frames)
        sr = sr[..., slice_h, slice_w].cpu().detach()
        if labels is not None:
            targets = torch.split(labels[0], 1, dim=1)
            targets = [t.squeeze(1) for t in targets]
            hr = targets[self.depth // 2]
            metrics['psnr'] = psnr(sr, hr)
            writer = get_writer(self.name)
            if writer is not None:
                step = kwargs['epoch']
                writer.image('clean', sr.clamp(0, 1), step=step)
                writer.image('warp/0', warps[0].clamp(0, 1), step=step)
                writer.image('warp/1', warps[-1].clamp(0, 1), step=step)
        return [sr.numpy()], metrics
