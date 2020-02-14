#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 11

import numpy as np
import torch
import torch.nn.functional as F

from VSR.Util.Math import gaussian_kernel, anisotropic_gaussian_kernel
from .Model import SuperResolution
from .srmd import ops, pca
from ..Framework.Summary import get_writer
from ..Util.Metrics import psnr
from ..Util.Utility import imfilter


class SRMD(SuperResolution):
  def __init__(self, scale, channel, degradation=None, **kwargs):
    super(SRMD, self).__init__(scale, channel)
    self.srmd = ops.Net(scale=scale, channels=channel, **kwargs)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    degradation = degradation or {}
    noise = degradation.get('noise', 0)
    if noise > 1:
      noise /= 255
    assert 0 <= noise <= 1
    self.pca_dim = kwargs.get('pca_dim', pca._PCA.shape[0])
    self.kernel_size = degradation.get('kernel_size', 15)
    self.ktype = degradation.get('kernel_type', 'isotropic')
    self.l1 = degradation.get('l1', 0.1)
    self.l2 = degradation.get('l2', 0.1)
    self.theta = degradation.get('theta', 0.1)
    self.noise = noise

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
    l1 = np.random.uniform(0.1, 10)
    l2 = np.random.uniform(0.1, l1)
    return self.gen_kernel('anisotropic', self.kernel_size, l1, l2, theta)

  def gen_random_noise(self, shape):
    stddev = np.random.uniform(0, 75 / 255, size=[shape[0]])
    noise = np.random.normal(size=shape) * stddev
    return noise, stddev

  def train(self, inputs, labels, learning_rate=None):
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    lr = inputs[0]
    batch = lr.shape[0]
    noise, stddev = self.gen_random_noise(lr.shape)
    kernel = [self.gen_random_kernel() for _ in range(batch)]
    degpar = torch.tensor([pca.get_degradation(k) for k in kernel],
                          dtype=lr.dtype, device=lr.device)
    kernel = torch.tensor(kernel, dtype=lr.dtype, device=lr.device)
    noise = torch.tensor(noise, dtype=lr.dtype, device=lr.device)
    stddev = torch.tensor(stddev, dtype=lr.dtype, device=lr.device)
    lr = imfilter(lr, kernel) + noise
    sr = self.srmd(lr, degpar, stddev)
    loss = F.l1_loss(sr, labels[0])
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {
      'loss': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    lr = inputs[0]
    batch = lr.shape[0]
    degpar = torch.tensor(
        [
          pca.get_degradation(self.gen_kernel(self.ktype,
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
    sr = self.srmd(lr, degpar, stddev).detach().cpu()
    if labels is not None:
      metrics['psnr'] = psnr(sr, labels[0])
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs.get('epoch', 0)
        writer.image('gt', labels[0], step=step)
        writer.image('clean', sr.clamp(0, 1), step=step)
    return [sr.numpy()], metrics

  def export(self, export_dir):
    """An example of how to export ONNX format"""

    # ONNX needs input placeholder to export model!
    # Sounds stupid to set a 48x48 inputs.

    device = list(self.srmd.parameters())[0].device
    inputs = torch.randn(1, self.channel, 144, 128, device=device)
    pca = torch.randn(1, self.pca_dim, 1, device=device)
    noise = torch.randn(1, 1, device=device)
    torch.onnx.export(self.srmd, (inputs, pca, noise), export_dir / 'srmd.onnx')
