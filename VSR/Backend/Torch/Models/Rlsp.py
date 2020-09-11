"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-8-5

RLSP
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..Framework.Summary import get_writer
from ..Util import psnr, upsample
from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d
from .Ops.Scale import SpaceToDepth

LOG = logging.getLogger("VSR.RLSP")
LOG.info("LICENSE: RLSP is proposed by D. Fuoli, et. al. "
         "implemented by LoSeall. "
         "@loseall https://github.com/loseall/VideoSuperResolution")


def get_zero_control_vec(batch, device):
    x = torch.zeros(batch, 7, dtype=torch.float32, device=device)
    x[:, 0] = 0.0
    x[:, 1] = 0.0
    x[:, 2] = 0.0
    x[:, 3] = 0.0
    x[:, 4] = 0.0
    x[:, 5] = 0.0
    x[:, 6] = 0.1
    return x


class Shift(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shift, self).__init__()
        self.alpha = nn.Sequential(nn.Linear(in_channels, out_channels),
                                   nn.Sigmoid())
        self.beta = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Tanh())

    def forward(self, x, w):
        a = self.alpha(w).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(w).unsqueeze(-1).unsqueeze(-1)
        return x * a + b


class Shifter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shifter, self).__init__()
        self.alpha = nn.Sequential(nn.Linear(in_channels, out_channels),
                                   nn.Sigmoid())
        self.beta = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Tanh())

    def forward(self, w):
        a = self.alpha(w).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(w).unsqueeze(-1).unsqueeze(-1)
        return a, b


class Mlp(nn.Module):
    def __init__(self, b1, b2, f1, f2, mlp_width, mlp_depth):
        super(Mlp, self).__init__()
        mlp = [nn.Linear(7, mlp_width), nn.ReLU(inplace=True)]
        for _ in range(1, mlp_depth):
            mlp += [nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*mlp)
        shift = []
        for _ in range(b1):
            shift.append(Shifter(mlp_width, f1))
        for _ in range(b2):
            shift.append(Shifter(mlp_width, f2))
        self.shift = nn.Sequential(*shift)

    def forward(self, z):
        w = self.mlp(z)
        r = []
        for m in self.shift:
            r.append(m(w))
        return tuple(r)


class RLSPCell(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, layers):
        super(RLSPCell, self).__init__()
        mlp = [nn.Linear(3, 32), nn.ReLU(inplace=True)]
        for _ in range(1, 32):
            mlp += [nn.Linear(32, 32), nn.ReLU(inplace=True)]
        self.mpl = nn.Sequential(*mlp)
        cell = [EasyConv2d(in_channels, hidden_channels, 3, activation='relu')]
        shift = []
        for i in range(1, layers - 1):
            cell.append(
                EasyConv2d(hidden_channels, hidden_channels, 3, activation='relu'))
            shift.append(Shift(32, hidden_channels))
        self.cell = nn.Sequential(*cell)
        self.hidden = EasyConv2d(hidden_channels, hidden_channels, 3,
                                 activation='relu')
        self.exit = EasyConv2d(hidden_channels, out_channels, 3)

    def forward(self, lr_frames, feedback, hidden_state, z):
        lr = torch.cat(lr_frames, dim=1)
        w = self.mpl(z)
        inputs = torch.cat((lr, hidden_state, feedback), dim=1)
        x = self.cell[0](inputs)
        for s, c in zip(self.shift, self.cell[1:]):
            x = s(x, w)
            x = c(x)
        next_hidden_state = self.hidden(x)
        residual = self.exit(x)
        return residual, next_hidden_state


class RlspNet(nn.Module):
    def __init__(self, scale, channel, depth=3, layers=7, filters=64):
        super(RlspNet, self).__init__()
        in_channels = channel * depth + filters + channel * scale ** 2
        self.rlspcell = RLSPCell(
            in_channels, channel * scale ** 2, filters, layers)
        self.shuffle = SpaceToDepth(scale)
        self.f = filters
        self.d = depth
        self.s = scale

    def forward(self, lr, sr, hidden, z=None):
        if hidden is None:
            shape = list(lr[0].shape)
            shape[1] = self.f
            hidden = torch.zeros(*shape, device=lr[0].device)
        if z is None:
            shape = list(lr[0].shape)
            z = get_zero_control_vec(shape[0], lr[0].device)
        center = F.interpolate(lr[self.d // 2], scale_factor=self.s)
        feedback = self.shuffle(sr).detach()
        res, next = self.rlspcell(lr, feedback, hidden, z)
        out = center + F.pixel_shuffle(res, self.s)
        return out, next


class RLSP(SuperResolution):
    """
    Args:
      clips: how many adjacent LR frames to use
      layers: number of convolution layers in RLSP cell
      filters: number of convolution filters

    Note:
      `depth` represents total sequences to train and evaluate
    """

    def __init__(self, scale, channel, clips=3, layers=7, filters=64, **kwargs):
        super(RLSP, self).__init__(scale=scale, channel=channel)
        self.rlsp = RlspNet(scale, channel, clips, layers, filters)
        self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
        self.clips = clips

    def train(self, inputs, labels, learning_rate=None):
        frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
        labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
        if learning_rate:
            for param_group in self.adam.param_groups:
                param_group["lr"] = learning_rate
        total_loss = 0
        image_loss = 0
        last_hidden = None
        last_sr = upsample(frames[0], self.scale)
        last_sr = torch.zeros_like(last_sr)
        for i in range(self.clips // 2, len(frames) - self.clips // 2):
            lr_group = [frames[i - self.clips // 2 + j]
                        for j in range(self.clips)]
            sr, hidden = self.rlsp(lr_group, last_sr, last_hidden)
            last_hidden = hidden
            last_sr = sr.detach()
            l2_image = F.mse_loss(sr, labels[i])
            loss = l2_image
            total_loss += loss
            image_loss += l2_image.detach()
        self.adam.zero_grad()
        total_loss.backward()
        self.adam.step()
        return {
            'total_loss': total_loss.detach().cpu().numpy() / len(frames),
            'image_loss': image_loss.cpu().numpy() / len(frames),
        }

    def eval(self, inputs, labels=None, **kwargs):
        metrics = {}
        frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
        if labels is not None:
            labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
        psnr_values = []
        predicts = []
        last_sr = upsample(frames[0], self.scale)
        last_sr = torch.zeros_like(last_sr)
        last_hidden = None
        i = 0
        for i in range(self.clips // 2, len(frames) - self.clips // 2):
            lr_group = [frames[i - self.clips // 2 + j]
                        for j in range(self.clips)]
            sr, _ = self.rlsp(lr_group, last_sr, last_hidden)
            last_sr = sr.detach()
            predicts.append(sr.cpu().detach().numpy())
            if labels is not None:
                psnr_values.append(psnr(sr, labels[i]))
        metrics['psnr'] = np.mean(psnr_values)
        writer = get_writer(self.name)
        if writer is not None:
            step = kwargs['epoch']
            writer.image('clean', last_sr.clamp(0, 1), step=step)
            writer.image('label', labels[i], step=step)
        return predicts, metrics
