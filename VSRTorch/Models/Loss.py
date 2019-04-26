#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/9 下午2:41

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
