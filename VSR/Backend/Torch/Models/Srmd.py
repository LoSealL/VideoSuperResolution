"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-11

SRMD
"""
import logging

import numpy as np
import torch
import torch.nn as nn

from VSR.Util.Math import anisotropic_gaussian_kernel, gaussian_kernel
from VSR.Util.PcaPrecompute import get_degradation

from ..Util import imfilter
from .Ops.Blocks import EasyConv2d
from .Ops.Discriminator import DCGAN
from .Optim.SISR import PerceptualOptimizer

logging.getLogger("VSR.SRFEAT").info(
    "LICENSE: SRMD is proposed by Kai Zhang, et. al. "
    "Implemented via PyTorch by @LoSealL.")


class Net(nn.Module):
    """
    SRMD CNN network. 12 conv layers
    """

    def __init__(self, scale=4, channels=3, layers=12, filters=128,
                 pca_length=15):
        super(Net, self).__init__()
        self.pca_length = pca_length
        net = [EasyConv2d(channels + pca_length + 1,
                          filters, 3, activation='relu')]
        net += [EasyConv2d(filters, filters, 3, activation='relu') for _ in
                range(layers - 2)]
        net += [EasyConv2d(filters, channels * scale ** 2, 3),
                nn.PixelShuffle(scale)]
        self.body = nn.Sequential(*net)

    def forward(self, x, kernel=None, noise=None):
        if kernel is None and noise is None:
            kernel = torch.zeros(
                x.shape[0], 15, 1, device=x.device, dtype=x.dtype)
            noise = torch.zeros(
                x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        # degradation parameter
        degpar = torch.cat([kernel, noise.reshape([-1, 1, 1])], dim=1)
        degpar = degpar.reshape([-1, 1 + self.pca_length, 1, 1])
        degpar = torch.ones_like(x)[:, 0:1] * degpar
        _x = torch.cat([x, degpar], dim=1)
        return self.body(_x)


class SRMD(PerceptualOptimizer):
    def __init__(self, scale, channel, degradation=None, layers=12, filters=128,
                 pca_length=15, **kwargs):
        degradation = degradation or {}
        noise = degradation.get('noise', 0)
        if noise > 1:
            noise /= 255
        assert 0 <= noise <= 1
        self.pca_dim = kwargs.get('pca_dim', 15)
        self.kernel_size = degradation.get('kernel_size', pca_length)
        self.ktype = degradation.get('kernel_type', 'isotropic')
        self.l1 = degradation.get('l1', 0.1)
        self.l2 = degradation.get('l2', 0.1)
        self.theta = degradation.get('theta', 0.1)
        self.noise = noise
        self.blur_padding = torch.nn.ReflectionPad2d(7)
        self.srmd = Net(scale, channel, layers, filters, pca_length)
        disc_opt = {
            'channel': channel, 'num_layers': 10, 'scale': scale, 'norm': 'BN'
        }
        super(SRMD, self).__init__(scale, channel, discriminator=DCGAN,
                                   discriminator_kwargs=disc_opt, **kwargs)

    def gen_kernel(self, ktype, ksize, l1, l2=None, theta=0):
        if ktype == 'isotropic':
            kernel = gaussian_kernel(ksize, l1)
        elif ktype == 'anisotropic':
            kernel = anisotropic_gaussian_kernel(ksize, theta, l1, l2 or l1)
        else:
            # TODO(wenyi) this is "directKernel"
            raise NotImplementedError("DirectKernel not implemented.")
        return kernel

    def gen_random_kernel(self):
        theta = np.random.uniform(0, np.pi)
        l1 = np.random.uniform(0.1, self.l1)
        l2 = np.random.uniform(0.1, l1)
        return self.gen_kernel('anisotropic', self.kernel_size, l1, l2, theta)

    def gen_random_noise(self, shape):
        stddev = np.random.uniform(0, self.noise, size=[shape[0]])
        noise = np.random.normal(size=shape) * stddev.reshape([-1, 1, 1, 1])
        return noise, stddev

    def fn(self, lr):
        batch = lr.shape[0]
        if self.srmd.training:
            noise, stddev = self.gen_random_noise(lr.shape)
            kernel = [self.gen_random_kernel() for _ in range(batch)]
            degpar = torch.tensor([get_degradation(k) for k in kernel],
                                  dtype=lr.dtype, device=lr.device)
            kernel = torch.tensor(kernel, dtype=lr.dtype, device=lr.device)
            noise = torch.tensor(noise, dtype=lr.dtype, device=lr.device)
            stddev = torch.tensor(stddev, dtype=lr.dtype, device=lr.device)
            lr = imfilter(lr, kernel, self.blur_padding) + noise
            sr = self.srmd(lr, degpar, stddev)
        else:
            degpar = torch.tensor(
                [
                    get_degradation(self.gen_kernel(self.ktype,
                                                    self.kernel_size,
                                                    self.l1,
                                                    self.l2,
                                                    self.theta))
                ] * batch,
                dtype=lr.dtype,
                device=lr.device)
            stddev = torch.tensor(
                [self.noise] * batch,
                dtype=lr.dtype,
                device=lr.device)
            sr = self.srmd(lr, degpar, stddev)
        return sr

    def export(self, export_dir):
        """An example of how to export ONNX format"""

        # ONNX needs input placeholder to export model!
        # Sounds stupid to set a 48x48 inputs.

        device = list(self.srmd.parameters())[0].device
        inputs = torch.randn(1, self.channel, 144, 128, device=device)
        pca = torch.randn(1, self.pca_dim, 1, device=device)
        noise = torch.randn(1, 1, device=device)
        torch.onnx.export(self.srmd, (inputs, pca, noise),
                          export_dir / 'srmd.onnx')
