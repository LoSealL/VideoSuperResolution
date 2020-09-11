"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-2

SOF-VSR
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Util import Metrics
from ..Util.Metrics import total_variance
from .Model import SuperResolution
from .Ops.Motion import STN

_logger = logging.getLogger("VSR.SOF")
_logger.info("LICENSE: SOF-VSR is implemented by Longguan Wang. "
             "@LongguanWang https://github.com/LongguangWang/SOF-VSR.")


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
        self.warper = STN()

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
        # Level 2
        x_L2 = self.warper(x[:, 0, :, :].unsqueeze(1), optical_flow_L1_upscaled,
                           gain=16)
        x_L2_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L2
        x_L2 = torch.cat((x, x_L2, x_L2_res, optical_flow_L1_upscaled), 1)
        input_L2 = self.conv_L2_1(x_L2)
        buffer_1 = self.RDB2_1(input_L2)
        buffer_2 = self.RDB2_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L2 = self.bottleneck_L2(buffer)
        optical_flow_L2 = self.conv_L2_2(optical_flow_L2)
        optical_flow_L2 = optical_flow_L2 + optical_flow_L1_upscaled
        # Level 3
        x_L3 = self.warper(torch.unsqueeze(x[:, 0, :, :], dim=1),
                           optical_flow_L2, gain=16)
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


class Sofvsr(nn.Module):
    def __init__(self, scale, channel, depth):
        super(Sofvsr, self).__init__()
        self.upscale_factor = scale
        self.c = channel
        self.OFRnet = OFRnet(upscale_factor=scale)
        self.SRnet = SRnet(scale, channel, depth)
        self.warper = STN()

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
                draft_01 = self.warper(x[:, :self.c, :, :],
                                       flow_01_L3[:, :, i::self.upscale_factor,
                                                  j::self.upscale_factor] / self.upscale_factor,
                                       gain=16)
                draft_21 = self.warper(x[:, self.c * 2:, :, :],
                                       flow_21_L3[:, :, i::self.upscale_factor,
                                                  j::self.upscale_factor] / self.upscale_factor,
                                       gain=16)
                draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)
        output = self.SRnet(draft_cube)
        return output, (flow_01_L3, flow_01_L2, flow_01_L1), (
            flow_21_L3, flow_21_L2, flow_21_L1)


class SOFVSR(SuperResolution):
    """Note: SOF is Y-channel SR with depth=3"""

    def __init__(self, scale, channel, depth=3, **kwargs):
        super(SOFVSR, self).__init__(scale, channel, **kwargs)
        self.sof = Sofvsr(scale, channel, depth)
        self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
        self.warper = STN()
        assert depth == 3
        self.center = depth // 2

    def train(self, inputs, labels, learning_rate=None):
        pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
        pre = torch.squeeze(pre, dim=1)
        cur = torch.squeeze(cur, dim=1)
        nxt = torch.squeeze(nxt, dim=1)
        low_res = torch.cat([pre, cur, nxt], dim=1)
        sr, flow01, flow21 = self.sof(low_res)
        hrp, hr, hrn = torch.split(labels[0], 1, dim=1)
        hrp = torch.squeeze(hrp, dim=1)
        hr = torch.squeeze(hr, dim=1)
        hrn = torch.squeeze(hrn, dim=1)
        loss_sr = F.mse_loss(sr, hr)
        pre_d = F.avg_pool2d(pre, 2)
        cur_d = F.avg_pool2d(cur, 2)
        nxt_d = F.avg_pool2d(nxt, 2)

        pre_d_warp = self.warper(pre_d, flow01[2], gain=16)
        pre_warp = self.warper(pre, flow01[1], gain=16)
        hrp_warp = self.warper(hrp, flow01[0], gain=16)
        nxt_d_warp = self.warper(nxt_d, flow21[2], gain=16)
        nxt_warp = self.warper(nxt, flow21[1], gain=16)
        hrn_warp = self.warper(hrn, flow21[0], gain=16)

        loss_lvl1 = F.mse_loss(pre_d_warp, cur_d) + F.mse_loss(nxt_d_warp, cur_d) + \
            0.01 * (total_variance(flow01[2]) + total_variance(flow21[2]))
        loss_lvl2 = F.mse_loss(pre_warp, cur) + F.mse_loss(nxt_warp, cur) + \
            0.01 * (total_variance(flow01[1]) + total_variance(flow21[1]))
        loss_lvl3 = F.mse_loss(hrp_warp, hr) + F.mse_loss(hrn_warp, hr) + \
            0.01 * (total_variance(flow01[0]) + total_variance(flow21[0]))
        loss = loss_sr + 0.01 * \
            (loss_lvl3 + 0.25 * loss_lvl2 + 0.125 * loss_lvl1)
        if learning_rate:
            for param_group in self.opt.param_groups:
                param_group["lr"] = learning_rate
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            'image': loss_sr.detach().cpu().numpy(),
            'flow/lvl1': loss_lvl1.detach().cpu().numpy(),
            'flow/lvl2': loss_lvl2.detach().cpu().numpy(),
            'flow/lvl3': loss_lvl3.detach().cpu().numpy(),
        }

    def eval(self, inputs, labels=None, **kwargs):
        metrics = {}
        pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
        low_res = torch.cat([pre, cur, nxt], dim=2)
        low_res = torch.squeeze(low_res, dim=1)
        sr, _, _ = self.sof(low_res)
        sr = sr.cpu().detach()
        if labels is not None:
            hr = labels[0][:, self.center]
            metrics['psnr'] = Metrics.psnr(sr, hr)
        return [sr.numpy()], metrics
