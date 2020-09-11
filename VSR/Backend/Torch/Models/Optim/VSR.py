"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

Common optimizer framework for video SR
"""

import torch

from ...Framework.Summary import get_writer
from ...Util import Metrics
from ...Util.Utility import pad_if_divide
from ..Model import SuperResolution
from .SISR import get_opt, get_pix_cri


class EarlyMergeOptimizer(SuperResolution):
    def __init__(self, scale, channel, **kwargs):
        super(EarlyMergeOptimizer, self).__init__(scale, channel)
        # gradient clip
        self.clip = kwargs.get('clip')
        # default use Adam with beta1=0.9 and beta2=0.999
        self.opt = get_opt(kwargs.get('opt'), self.trainable_variables(), 1e-4)
        self.padding = kwargs.get('padding', 0)
        self.pixel_cri = get_pix_cri(kwargs.get('cri_image'))

    def fn(self, x):
        raise NotImplementedError

    def flow(self, target, ref):
        raise NotImplementedError

    def warp(self, x, flow):
        raise NotImplementedError

    def train(self, inputs, labels, learning_rate=None):
        sr = self.fn(inputs[0])
        gt = labels[0][labels[0].shape[1] // 2]
        loss = self.pixel_cri(sr, gt)
        if learning_rate:
            for param_group in self.opt.param_groups:
                param_group["lr"] = learning_rate
        self.opt.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(
                self.trainable_variables(), self.clip)
        self.opt.step()
        return {'loss': loss.detach().cpu().numpy()}

    def eval(self, inputs, labels=None, **kwargs):
        metrics = {}
        _lr = inputs[0]
        if self.padding:
            lr = pad_if_divide(_lr, self.padding)
            a = lr.size(3) - _lr.size(3)
            b = lr.size(4) - _lr.size(4)
            slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
            slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
            sr = self.fn(lr)[..., slice_h, slice_w]
        else:
            sr = self.fn(_lr)
        sr = sr.cpu().detach()
        if labels is not None:
            gt = labels[0][labels[0].shape[1] // 2]
            metrics['psnr'] = Metrics.psnr(sr.numpy(), gt.cpu().numpy())
            writer = get_writer(self.name)
            if writer is not None:
                step = kwargs.get('epoch')
                writer.image('sr', sr.clamp(0, 1), max=1, step=step)
                writer.image('gt', gt, max=1, step=step)
        return [sr.numpy()], metrics
