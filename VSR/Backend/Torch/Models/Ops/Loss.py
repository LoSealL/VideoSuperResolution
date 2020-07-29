#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 16

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def total_variance(x, dims=(2, 3), reduction='mean'):
  tot_var = 0
  reduce = 1
  for dim in dims:
    row = x.split(1, dim=dim)
    reduce *= x.shape[dim]
    for i in range(len(row) - 1):
      tot_var += torch.abs(row[i] - row[i + 1]).sum()
  if reduction != 'mean':
    reduce = 1
  return tot_var / reduce


def focal_l1_loss(x, y, a=2, b=0, c=1, y_is_label=True, focal='edge'):
  if not y_is_label:
    x, y = y, x
  absolute_diff = torch.abs(x - y)
  if focal == 'edge':
    focal = F.pad(y[..., :-1, :-1] - y[..., 1:, 1:], [0, 1, 0, 1])
  elif focal == 'focal':
    focal = absolute_diff
  focal = torch.abs((focal - focal.mean()) / focal.std())
  focal = (focal + 1) / 2
  tuned_diff = torch.pow(focal, a) * absolute_diff
  loss = b * absolute_diff + c * tuned_diff
  return loss.mean()


def gan_bce_loss(x, as_real: bool):
  """vanilla GAN binary cross entropy loss"""
  if as_real:
    return F.binary_cross_entropy_with_logits(x, torch.ones_like(x))
  else:
    return F.binary_cross_entropy_with_logits(x, torch.zeros_like(x))


def rgan_bce_loss(x, y, x_real_than_y: bool = True):
  """relativistic GAN loss"""
  if x_real_than_y:
    return F.binary_cross_entropy_with_logits(x - y, torch.ones_like(x))
  else:
    return F.binary_cross_entropy_with_logits(y - x, torch.ones_like(x))


def ragan_bce_loss(x, y, x_real_than_y: bool = True):
  """relativistic average GAN loss"""
  if x_real_than_y:
    return F.binary_cross_entropy_with_logits(x - y.mean(),
                                              torch.ones_like(x)) + \
           F.binary_cross_entropy_with_logits(y - x.mean(),
                                              torch.zeros_like(y))
  else:
    return F.binary_cross_entropy_with_logits(y - x.mean(),
                                              torch.ones_like(x)) + \
           F.binary_cross_entropy_with_logits(x - y.mean(),
                                              torch.zeros_like(y))


class GeneratorLoss(nn.Module):
  def __init__(self, name='GAN'):
    self.type = name
    super(GeneratorLoss, self).__init__()

  def forward(self, x, y=None):
    if self.type == 'RGAN':
      return rgan_bce_loss(x, y, True)
    elif self.type == 'RAGAN':
      return ragan_bce_loss(x, y, True)
    else:
      return gan_bce_loss(x, True)


class DiscriminatorLoss(nn.Module):
  def __init__(self, name='GAN'):
    self.type = name
    super(DiscriminatorLoss, self).__init__()

  def forward(self, x, y=None):
    if self.type == 'RGAN':
      return rgan_bce_loss(x, y, False)
    elif self.type == 'RAGAN':
      return ragan_bce_loss(x, y, False)
    else:
      return gan_bce_loss(x, False) + gan_bce_loss(y, True)


class VggFeatureLoss(nn.Module):
  # layer name stick to keras model
  _LAYER_NAME = {
    'block1_conv1': 1,
    'block1_conv2': 3,
    'block2_conv1': 6,
    'block2_conv2': 8,
    'block3_conv1': 11,
    'block3_conv2': 13,
    'block3_conv3': 15,
    'block3_conv4': 17,
    'block4_conv1': 20,
    'block4_conv2': 22,
    'block4_conv3': 24,
    'block4_conv4': 26,
    'block5_conv1': 29,
    'block5_conv2': 31,
    'block5_conv3': 33,
    'block5_conv4': 35,
  }
  """VGG19 based perceptual loss from ECCV 2016.
  
  Args:
    layer_names: a list of `_LAYER_NAME` strings, specify features to forward.
    before_relu: forward features before ReLU activation.
    external_weights: a path to an external vgg weights file, default download
      from model zoo.
  """

  def __init__(self, layer_names, before_relu=False, external_weights=None):
    super(VggFeatureLoss, self).__init__()
    if not external_weights:
      net = torchvision.models.vgg19(pretrained=True)
    else:
      net = torchvision.models.vgg19()
      # TODO map_location=?
      net.load_state_dict(torch.load(external_weights))
    for p in net.parameters():
      p.requires_grad = False
    self.childs = nn.Sequential(*net.features.children())
    self.eval()
    self.exit_id = [self._LAYER_NAME[n] - int(before_relu) for n in layer_names]

  def normalize(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # see https://pytorch.org/docs/master/torchvision/models.html for details
    assert x.size(1) == 3, "wrong channel! must be 3!!"
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    mean = mean.to(x.device)
    std = std.to(x.device)
    return (x - mean) / std

  def forward(self, x):
    exits = []
    x = self.normalize(x)
    for i, fn in enumerate(self.childs.children()):
      x = fn(x)
      if i in self.exit_id:
        exits.append(x)
      if i >= max(self.exit_id):
        break
    return exits
