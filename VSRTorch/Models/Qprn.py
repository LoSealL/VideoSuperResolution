#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/8 下午2:43

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .Arch import CascadeRdn, Rdb, SpaceToDepth, Upsample
from .Crdn import Upsample as RsrUp
from .Discriminator import DCGAN
from .Loss import gan_bce_loss, total_variance
from .Model import SuperResolution
from .video.motion import STTN
from ..Framework.Summary import get_writer
from ..Framework.Trainer import SRTrainer, from_tensor, to_tensor
from ..Util import Metrics
from ..Util.Utility import pad_if_divide


class Fnet(nn.Module):
  def __init__(self, channel, L=2, gain=64):
    super(Fnet, self).__init__()
    self.lq_entry = nn.Sequential(
      nn.Conv2d(channel * (L + 1), 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      Rdb(64), Rdb(64))
    self.hq_entry = nn.Sequential(
      nn.Conv2d(channel * L, 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      Rdb(64), Rdb(64))
    self.flownet = nn.Sequential(
      nn.Conv2d(128, 64, 1, 1, 0),
      Rdb(64), Rdb(64), Upsample(64, 4),
      nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())
    gain = torch.as_tensor([L, gain, gain], dtype=torch.float32)
    self.gain = gain.reshape(1, 3, 1, 1)

  def forward(self, lq, hq):
    x = torch.cat(lq, dim=1)
    y = torch.cat(hq, dim=1)
    x = self.lq_entry(x)
    y = self.hq_entry(y)
    z = torch.cat([x, y], dim=1)
    flow = self.flownet(z)
    gain = self.gain.to(flow.device)
    return flow * gain


class Unet(nn.Module):
  def __init__(self, channel, N=2):
    super(Unet, self).__init__()
    self.entry = nn.Sequential(
      nn.Conv2d(channel * N, 32, 3, 1, 1),
      SpaceToDepth(2),
      nn.Conv2d(128, 32, 1, 1, 0))
    self.exit = nn.Sequential(
      Upsample(32, 2), nn.Conv2d(32, channel, 3, 1, 1))
    self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
    self.up1 = RsrUp([64, 32])
    self.cb = CascadeRdn(64, 3, True)

  def forward(self, *inputs):
    inp = torch.cat(inputs, dim=1)  # w
    c0 = self.entry(inp)  # w / 2
    c1 = self.down1(c0)  # w / 4
    x = self.cb(c1)  # w / 4
    c2 = self.up1(x, c0)
    out = self.exit(c2)
    return out


class Composer(nn.Module):
  def __init__(self, channel, L=2, gain=64):
    super(Composer, self).__init__()
    self.flownet = Fnet(channel, L, gain=gain)
    self.refiner = Unet(channel, N=2)
    self.warpper = STTN((0, 2, 1, 3, 4), padding_mode='border')
    self.L = L

  def forward(self, lq, hq):
    """
    :param lq: [lq_{i+1}, lq_i, lq_{i-1}]
    :param hq: [hq_i, hq_{i-1}]
    :return: hq_{i+1}
    """
    assert isinstance(lq, list) and isinstance(hq, list)
    assert len(lq) == self.L + 1 and len(hq) == self.L
    flow = self.flownet(lq, hq)
    d, u, v = [t.squeeze(1) for t in flow.split(1, dim=1)]
    warp = self.warpper(torch.stack(hq, dim=1), d, u, v)
    warp = warp.squeeze(1)
    y = self.refiner(warp, lq[0])
    return y, warp, (d, u, v)


class QPRN(SuperResolution):
  """QP-based video restoration network"""

  def __init__(self, gain, scale, channel, **kwargs):
    super(QPRN, self).__init__(scale, channel, **kwargs)
    self.debug = kwargs.get('debug', {})
    # image, flow, tv, history, gan
    self.w = kwargs.get('weights', [1, 10, 1.0e-4, 0.1, 5e-3])
    self.qprn = Composer(channel, L=2, gain=gain)
    self.adam = torch.optim.Adam(self.trainable_variables('qprn'), 1e-4)
    if self.debug.gan:
      self.dnet = DCGAN(channel * 4, 9, 'bn', 'A')
      self.adam_d = torch.optim.Adam(self.trainable_variables('dnet'), 1e-4)
    self._trainer = _Trainer

  def train(self, inputs, labels, learning_rate=None):
    metrics = {}
    frames = list(torch.split(inputs[0], 1, dim=1))
    labels = list(torch.split(labels[0], 1, dim=1))
    total_loss = 0
    flow_loss = 0
    image_loss = 0
    his_loss = 0
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    self.adam.zero_grad()
    # # time extension
    # frames_rev, labels_rev = frames.copy(), labels.copy()
    # frames_rev.reverse()
    # labels_rev.reverse()
    # rev_windows = {'lq': [], 'hq': [], 'label': []}
    # warps = []
    # for lq, label in zip(frames_rev, labels_rev):
    #   lq = lq.squeeze(1)
    #   label = label.squeeze(1)
    #   if not rev_windows['lq']:
    #     rev_windows['lq'] = [lq.detach(), lq.detach(), lq.detach()]
    #     rev_windows['hq'] = [lq.detach(), lq.detach()]
    #     rev_windows['label'] = [label.detach(), label.detach(), label.detach()]
    #   rev_windows['lq'].pop(-1)
    #   rev_windows['lq'].insert(0, lq.detach())
    #   rev_windows['label'].pop(-1)
    #   rev_windows['label'].insert(0, label.detach())
    #   hq, hq_warp, _ = self.qprn(rev_windows['lq'], rev_windows['hq'])
    #   warps.append(hq_warp)
    #   rev_windows['hq'].pop(-1)
    #   rev_windows['hq'].insert(0, hq.detach())
    # warps.reverse()
    idr_lq = frames[0].squeeze(1)
    idr_hq = labels[0].squeeze(1)
    idr = self.qprn.refiner(idr_lq, idr_lq)
    length = self.qprn.L + 1
    windows = {
      'lq': [idr_lq.detach() for _ in range(length)],
      'hq': [idr.detach() for _ in range(1, length)],
      'label': [idr_hq.detach() for _ in range(length)]
    }
    history = [idr.detach()]
    history_hq = [idr_hq.detach()]
    for lq, label in zip(frames[1:], labels[1:]):
      lq = lq.squeeze(1)
      label = label.squeeze(1)
      windows['lq'].pop(-1)
      windows['lq'].insert(0, lq.detach())
      windows['label'].pop(-1)
      windows['label'].insert(0, label.detach())
      hq, hq_warp, flow = self.qprn(windows['lq'], windows['hq'])
      lq_to_warp = windows['lq'][1:]
      lb_to_warp = windows['label'][1:]
      windows['hq'].pop(-1)
      windows['hq'].insert(0, hq.detach())
      lq_warp = self.qprn.warpper(torch.stack(lq_to_warp, dim=1), *flow)
      lq_warp = lq_warp.squeeze(1)
      lb_warp = self.qprn.warpper(torch.stack(lb_to_warp, dim=1), *flow)
      lb_warp = lb_warp.squeeze(1)

      history.append(hq.detach())
      history_hq.append(label.detach())
      l2_his = 0
      for his, his_hq in zip(history[:-1], history_hq[:-1]):
        his_to_warp = [his.detach() for _ in range(1, length)]
        his_flow = self.qprn.flownet(his_to_warp + [hq.detach()], his_to_warp)
        his_flow = [t.squeeze(1) for t in his_flow.split(1, dim=1)]
        his_warp = self.qprn.warpper(torch.stack(his_to_warp, dim=1), *his_flow)
        his_warp = his_warp.squeeze(1)
        his_lb_to_warp = [his_hq.detach() for _ in range(1, length)]
        his_lb_warp = self.qprn.warpper(torch.stack(his_lb_to_warp, dim=1),
                                        *his_flow).squeeze(1)
        his_mask = torch.exp(-50.0 * (his_lb_warp - label) ** 2)
        l2_his += torch.mean(his_mask * (his_warp - hq) ** 2)

      l1_image = F.l1_loss(hq, label)
      l2_warp = 0.8 * F.mse_loss(lb_warp, label) + 0.2 * F.mse_loss(lq_warp, lq)
      tv_flow = total_variance(torch.stack(flow, dim=1), dims=(2, 3))
      loss = self.w[0] * l1_image + self.w[1] * l2_warp + self.w[2] * tv_flow
      loss += self.w[3] * l2_his
      if self.debug.gan:
        fake = self.dnet(torch.cat((hq, hq_warp.detach(),
                                    lq, lq_warp.detach()), dim=1))
        gloss = gan_bce_loss(fake, True)
        loss += gloss * self.w[4]
        metrics['g'] = gloss.detach().cpu().numpy()
      self.adam.zero_grad()
      loss.backward()
      self.adam.step()
      total_loss += loss.detach()
      image_loss += l1_image.detach()
      flow_loss += l2_warp.detach()
      his_loss += l2_his.detach()
      if self.debug.gan:
        fake = self.dnet(torch.cat((hq.detach(), hq_warp.detach(),
                                    lq, lq_warp.detach()), dim=1))
        real = self.dnet(torch.cat((label, lb_warp.detach(),
                                    lq, lq_warp.detach()), dim=1))
        d_loss = gan_bce_loss(fake, False) + gan_bce_loss(real, True)
        self.adam_d.zero_grad()
        d_loss.backward()
        self.adam_d.step()
        metrics['d'] = d_loss.detach().cpu().numpy()
    metrics.update({
      'total_loss': total_loss.detach().cpu().numpy() / len(frames),
      'image_loss': image_loss.detach().cpu().numpy() / len(frames),
      'flow_loss': flow_loss.detach().cpu().numpy() / len(frames),
      'his_loss': his_loss.detach().cpu().numpy() / len(frames),
    })
    return metrics

  def eval(self, inputs, labels=None, **kwargs):
    frames = torch.split(inputs[0], 1, dim=1)
    metrics = {}
    idr_lq_ = frames[0].squeeze(1)
    idr_lq = pad_if_divide(idr_lq_, 4, 'reflect')
    a = idr_lq.shape[-2] - idr_lq_.shape[-2]
    c = idr_lq.shape[-1] - idr_lq_.shape[-1]
    a, b = a // 2, -a // 2
    c, d = c // 2, -c // 2
    if a == 0: a = b = None
    if c == 0: c = d = None
    idr = self.qprn.refiner(idr_lq, idr_lq)
    length = self.qprn.L + 1
    windows = {
      'lq': [idr_lq.detach() for _ in range(length)],
      'hq': [idr.detach() for _ in range(1, length)],
      'predict': [idr.detach().cpu().numpy()[..., a:b, c:d]]
    }
    time_loss = 0
    for lq_ in frames[1:]:
      lq_ = lq_.squeeze(1)
      lq = pad_if_divide(lq_, 4, 'reflect')
      windows['lq'].pop(-1)
      windows['lq'].insert(0, lq.detach())
      hq, hq_warp, flow = self.qprn(windows['lq'], windows['hq'])
      windows['hq'].pop(-1)
      windows['hq'].insert(0, hq.detach())
      lq_to_warp = windows['lq'][1:]
      lq_warp = self.qprn.warpper(torch.stack(lq_to_warp, dim=1), *flow)
      lq_warp = lq_warp.squeeze(1)
      if self.debug.get('see_warp'):
        windows['predict'].append(hq_warp.detach().cpu().numpy()[..., a:b, c:d])
      elif self.debug.get('see_flow'):
        windows['predict'].append(torch.stack(
          flow[1:], dim=1).detach().cpu().numpy()[..., a:b, c:d])
      else:
        windows['predict'].append(hq.detach().cpu().numpy()[..., a:b, c:d])
      time_loss += F.mse_loss(hq, hq_warp).detach()
    if labels is not None:
      targets = torch.split(labels[0], 1, dim=1)
      targets = [t.squeeze(1) for t in targets]
      psnr = [Metrics.psnr(x, y) for x, y in zip(windows['predict'], targets)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hq', hq.clamp(0, 1), step=step)
        writer.image('lq', lq.clamp(0, 1), step=step)
        writer.image('idr', idr.clamp(0, 1), step=step)
        writer.image('warp', lq_warp.clamp(0, 1), step=step)
    metrics['time_loss'] = time_loss.detach().cpu().numpy() / len(frames)
    return windows['predict'], metrics


class _Trainer(SRTrainer):
  jitter = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage('RGB'),
    torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    torchvision.transforms.ToTensor()
  ])
  trans = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage('RGB'),
    torchvision.transforms.RandomAffine(0, [0.02, 0.02]),
    torchvision.transforms.ToTensor()
  ])

  def random_apply_fn(self, tensor, fn, prob=0.5):
    assert isinstance(tensor, torch.Tensor) and callable(fn)
    tensor_ = []
    for t in tensor.cpu():
      # enum batch
      p = np.random.rand()
      if p < prob:
        t = [fn(x.squeeze(0)) for x in t.split(1, dim=0)]
        t = torch.stack(t, dim=0)
      tensor_.append(t)
    tensor_ = torch.stack(tensor_, dim=0)
    if self.v.cuda and torch.cuda.is_available():
      tensor_ = tensor_.cuda()
    assert tensor_.shape == tensor.shape
    return tensor_

  def fn_train_each_step(self, label=None, feature=None, name=None, post=None):
    v = self.v
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    feature = to_tensor(feature, v.cuda)
    # Inline data augmentation
    # feature = self.random_apply_fn(feature, self.jitter, 0.3)
    # feature = self.random_apply_fn(feature, self.trans, 0.3)
    label = to_tensor(label, v.cuda)
    loss = self.model.train([feature], [label], v.lr)
    for _k, _v in loss.items():
      v.avg_meas[_k] = \
        v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
      loss[_k] = '{:08.5f}'.format(_v)
    v.loss = loss

  def fn_benchmark_each_step(self, label=None, feature=None, name=None,
                             post=None):
    v = self.v
    origin_feat = feature
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    feature = to_tensor(feature, v.cuda)
    label = to_tensor(label, v.cuda)
    with torch.set_grad_enabled(False):
      outputs, metrics = self.model.eval([feature], [label], epoch=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    outputs = [from_tensor(x) for x in outputs]
    for fn in v.output_callbacks:
      outputs = fn(outputs, input=origin_feat, label=label, name=name,
                   mode=v.color_format, subdir=v.subdir)
