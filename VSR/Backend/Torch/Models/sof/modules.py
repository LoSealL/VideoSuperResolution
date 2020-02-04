#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 22

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def optical_flow_warp(image, image_optical_flow):
  """
  Arguments
      image_ref: reference images tensor, (b, c, h, w)
      image_optical_flow: optical flow to image_ref (b, 2, h, w)
  """
  b, _, h, w = image.size()
  grid = np.meshgrid(range(w), range(h))
  grid = np.stack(grid, axis=-1).astype(np.float64)
  grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
  grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
  grid = grid.transpose(2, 0, 1)
  grid = np.tile(grid, (b, 1, 1, 1))
  grid = Variable(torch.Tensor(grid))
  if image_optical_flow.is_cuda == True:
    grid = grid.cuda()

  flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
  flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
  grid = grid + torch.cat((flow_0, flow_1), 1)
  grid = grid.transpose(1, 2)
  grid = grid.transpose(3, 2)
  output = F.grid_sample(image, grid, padding_mode='border')
  return output


class make_dense(nn.Module):
  def __init__(self, channels_in, channels_out, kernel_size=3):
    super(make_dense, self).__init__()
    self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2,
                          bias=False)

  def forward(self, x):
    out = self.leaky_relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class RDB(nn.Module):
  def __init__(self, nDenselayer, channels, growth):
    super(RDB, self).__init__()
    modules = []
    channels_buffer = channels
    for i in range(nDenselayer):
      modules.append(make_dense(channels_buffer, growth))
      channels_buffer += growth
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(channels_buffer, channels, kernel_size=1,
                              padding=0, bias=False)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out


class OFRnet(nn.Module):
  def __init__(self, upscale_factor):
    super(OFRnet, self).__init__()
    self.pool = nn.AvgPool2d(kernel_size=2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
    self.final_upsample = nn.Upsample(scale_factor=upscale_factor,
                                      mode='bilinear', align_corners=False)
    self.shuffle = nn.PixelShuffle(upscale_factor)
    self.upscale_factor = upscale_factor
    # Level 1
    self.conv_L1_1 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)
    self.RDB1_1 = RDB(4, 32, 32)
    self.RDB1_2 = RDB(4, 32, 32)
    self.bottleneck_L1 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
    self.conv_L1_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
    # Level 2
    self.conv_L2_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
    self.RDB2_1 = RDB(4, 32, 32)
    self.RDB2_2 = RDB(4, 32, 32)
    self.bottleneck_L2 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
    self.conv_L2_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
    # Level 3
    self.conv_L3_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
    self.RDB3_1 = RDB(4, 32, 32)
    self.RDB3_2 = RDB(4, 32, 32)
    self.bottleneck_L3 = nn.Conv2d(64, 2 * upscale_factor ** 2, 3, 1, 1,
                                   bias=False)
    self.conv_L3_2 = nn.Conv2d(2 * upscale_factor ** 2, 2 * upscale_factor ** 2,
                               3, 1, 1, bias=True)

  def forward(self, x):
    # Level 1
    x_L1 = self.pool(x)
    _, _, h, w = x_L1.size()
    input_L1 = self.conv_L1_1(x_L1)
    buffer_1 = self.RDB1_1(input_L1)
    buffer_2 = self.RDB1_2(buffer_1)
    buffer = torch.cat((buffer_1, buffer_2), 1)
    optical_flow_L1 = self.bottleneck_L1(buffer)
    optical_flow_L1 = self.conv_L1_2(optical_flow_L1)
    optical_flow_L1_upscaled = self.upsample(optical_flow_L1)  # *2
    # x_L1_res = optical_flow_warp(torch.unsqueeze(x_L1[:, 0, :, :], dim=1), optical_flow_L1) - torch.unsqueeze(x_L1[:, 1, :, :], dim=1)
    # Level 2
    x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1),
                             optical_flow_L1_upscaled)
    x_L2_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L2
    x_L2 = torch.cat((x, x_L2, x_L2_res, optical_flow_L1_upscaled), 1)
    input_L2 = self.conv_L2_1(x_L2)
    buffer_1 = self.RDB2_1(input_L2)
    buffer_2 = self.RDB2_2(buffer_1)
    buffer = torch.cat((buffer_1, buffer_2), 1)
    optical_flow_L2 = self.bottleneck_L2(buffer)
    optical_flow_L2 = self.conv_L2_2(optical_flow_L2)
    optical_flow_L2 = optical_flow_L2 + optical_flow_L1_upscaled
    # x_L2_res = optical_flow_warp(torch.unsqueeze(x_L2[:, 0, :, :], dim=1), optical_flow_L2) - torch.unsqueeze(x_L2[:, 1, :, :], dim=1)
    # Level 3
    x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1),
                             optical_flow_L2)
    x_L3_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L3
    x_L3 = torch.cat((x, x_L3, x_L3_res, optical_flow_L2), 1)
    input_L3 = self.conv_L3_1(x_L3)
    buffer_1 = self.RDB3_1(input_L3)
    buffer_2 = self.RDB3_2(buffer_1)
    buffer = torch.cat((buffer_1, buffer_2), 1)
    optical_flow_L3 = self.bottleneck_L3(buffer)
    optical_flow_L3 = self.conv_L3_2(optical_flow_L3)
    optical_flow_L3 = self.shuffle(optical_flow_L3) + self.final_upsample(
      optical_flow_L2)  # *4

    return optical_flow_L3, optical_flow_L2, optical_flow_L1


class SRnet(nn.Module):
  def __init__(self, s, c, d):
    """
    Args:
      s: scale factor
      c: channel numbers
      d: video sequence number
    """
    super(SRnet, self).__init__()
    self.conv = nn.Conv2d(c * (2 * s ** 2 + d), 64, 3, 1, 1, bias=False)
    self.RDB_1 = RDB(5, 64, 32)
    self.RDB_2 = RDB(5, 64, 32)
    self.RDB_3 = RDB(5, 64, 32)
    self.RDB_4 = RDB(5, 64, 32)
    self.RDB_5 = RDB(5, 64, 32)
    self.bottleneck = nn.Conv2d(384, c * s ** 2, 1, 1, 0, bias=False)
    self.conv_2 = nn.Conv2d(c * s ** 2, c * s ** 2, 3, 1, 1, bias=True)
    self.shuffle = nn.PixelShuffle(upscale_factor=s)

  def forward(self, x):
    input = self.conv(x)
    buffer_1 = self.RDB_1(input)
    buffer_2 = self.RDB_2(buffer_1)
    buffer_3 = self.RDB_3(buffer_2)
    buffer_4 = self.RDB_4(buffer_3)
    buffer_5 = self.RDB_5(buffer_4)
    output = torch.cat(
      (buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, input), 1)
    output = self.bottleneck(output)
    output = self.conv_2(output)
    output = self.shuffle(output)
    return output


class SOFVSR(nn.Module):
  def __init__(self, scale, channel, depth):
    super(SOFVSR, self).__init__()
    self.upscale_factor = scale
    self.c = channel
    self.OFRnet = OFRnet(upscale_factor=scale)
    self.SRnet = SRnet(scale, channel, depth)

  def forward(self, x):
    input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1),
                          torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)
    input_21 = torch.cat((torch.unsqueeze(x[:, 2, :, :], dim=1),
                          torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)
    flow_01_L3, flow_01_L2, flow_01_L1 = self.OFRnet(input_01)
    flow_21_L3, flow_21_L2, flow_21_L1 = self.OFRnet(input_21)
    draft_cube = x
    for i in range(self.upscale_factor):
      for j in range(self.upscale_factor):
        draft_01 = optical_flow_warp(x[:, :self.c, :, :],
                                     flow_01_L3[:, :, i::self.upscale_factor,
                                     j::self.upscale_factor] / self.upscale_factor)
        draft_21 = optical_flow_warp(x[:, self.c * 2:, :, :],
                                     flow_21_L3[:, :, i::self.upscale_factor,
                                     j::self.upscale_factor] / self.upscale_factor)
        draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)
    output = self.SRnet(draft_cube)
    return output, (flow_01_L3, flow_01_L2, flow_01_L1), (
      flow_21_L3, flow_21_L2, flow_21_L1)
