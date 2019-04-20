#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 4 - 16

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..edsr.common import MeanShift


class sub_pixel(nn.Module):
  def __init__(self, scale, act=False):
    super(sub_pixel, self).__init__()
    modules = []
    modules.append(nn.PixelShuffle(scale))
    self.body = nn.Sequential(*modules)  # nn.Sequential(*modules)

  def forward(self, x):
    x = self.body(x)
    return x


class RAN(nn.Module):
  def __init__(self, args):
    super(RAN, self).__init__()
    nChannel = args.n_colors
    nFeat = args.n_feats
    scale = args.scale[0]
    self.args = args

    self.plot_cnt = 0
    self.args = args
    self.id = [None] * 200
    # self.vis2 = visdom.Visdom(env='att_main')
    self.idx = 0

    rgb_mean = (0.45738, 0.43637, 0.40293)
    rgb_std = (1.0, 1.0, 1.0)
    self.sub_mean = MeanShift(255, rgb_mean, rgb_std, -1)
    self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1,
                           bias=True)
    self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1,
                           bias=True)
    self.Mod = torch.nn.ModuleList(
      [ST_Block5(args.n_resblocks, nFeat, i) for i in
       range(args.n_resgroups)])  # nn.ModuleList()
    self.spatial_att = Spatial_att_score2(nFeat, args.n_resgroups)
    # for i in range(10):
    #    self.Mod.append(ST_Block2(1,nFeat))
    # self.Mod = nn.Sequential(*self.Mod)
    # self.conv3 = nn.Conv2d(nFeat*10*2, nFeat, kernel_size=1, padding=0, bias=True, groups=2)
    self.G_F1 = nn.Conv2d(nFeat * args.n_resgroups, nFeat, kernel_size=1,
                          padding=0, bias=True)
    self.G_F33 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1,
                           bias=True)
    self.G_F3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
    self.GFF_3x3 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0,
                             bias=True)
    self.GFF_3x3_2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1,
                               bias=True)
    self.G_F2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

    self.Dblock = Dilated_block(nFeat)

    if scale == 4:
      self.up = nn.Sequential(
        nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1,
                  bias=True),
        sub_pixel(2),
        nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1,
                  bias=True),
        sub_pixel(2))
    else:
      self.up = nn.Sequential(
        nn.Conv2d(nFeat, nFeat * scale * scale, kernel_size=3,
                  padding=1, bias=True),
        sub_pixel(2))

    self.HR = nn.Sequential(
      nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
      nn.LeakyReLU(0.1),
      nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True))

  def forward(self, x):

    self.plot_cnt += 1

    i = x

    x_list = []
    out_list = []
    x = self.conv1(x)

    x = (self.conv2(x))  # ,5e-2)

    x_list.append(x)

    out_list.append(x)

    io = x
    # out_list.append(x)
    for ii, submodel in enumerate(self.Mod):
      if ii == 3:
        inp = x.detach()
      res, x, scale = submodel(x, x_list, out_list)
      if False:  # (ii+1)%4==0:
        x += io
        io = x
      out_list.append(res)
      x_list.append(x)

      ##for visdom
      if ii == 3:
        s = scale
    s = None
    # plot scale ------------------------------------------------------------------
    if self.plot_cnt == 200 and s is not None:
      self.plot_cnt = 0
      for idx, sle in enumerate(s):
        self.id[idx] = self.vis2.images(
          sle[0, :, :, :].unsqueeze(dim=1), nrow=6, win=self.id[idx],
          opts={'title': 'BLOCK-15'})

    x = self.G_F1(torch.cat(out_list[1:], dim=1))
    x = self.G_F33(x)

    if not self.args.nvis:
      self.id[self.idx] = self.vis2.heatmap(mask[0, 0, :, :],
                                            win=self.id[self.idx],
                                            opts={'title': 'SA'})
    self.idx += 1

    if self.idx == 20:
      self.idx = 0

    x = self.HR(x) + i

    return x


class Dilated_block(nn.Module):
  def __init__(self, inChannel):
    super(Dilated_block, self).__init__()

    self.conv11 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
    self.conv12 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=2,
                            dilation=2)

    self.conv21 = nn.Conv2d(inChannel, inChannel, kernel_size=5, padding=2)
    self.conv22 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=3,
                            dilation=3)

    self.conv31 = nn.Conv2d(inChannel, inChannel, kernel_size=7, padding=3)
    self.conv32 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=3,
                            dilation=3)

    self.Fuse = nn.Conv2d(inChannel * 3, inChannel, kernel_size=1,
                          padding=0)

  def forward(self, x):
    i = x
    x1 = self.conv12(F.leaky_relu(self.conv11(x), 0.1))
    x2 = self.conv22(F.leaky_relu(self.conv21(x), 0.1))
    x3 = self.conv32(F.leaky_relu(self.conv31(x), 0.1))

    x = self.Fuse(torch.cat([x1, x2, x3], dim=1))

    return x + i


class Spatial_att_score2(nn.Module):
  def __init__(self, inChannel, num_block):
    super(Spatial_att_score2, self).__init__()

    self.conv1 = nn.Conv2d(inChannel * num_block, inChannel, kernel_size=1,
                           padding=0)
    self.conv2 = nn.Conv2d(inChannel * 2, inChannel, kernel_size=3,
                           padding=1)
    self.conv3 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(inChannel, 1, kernel_size=3, padding=1)

    self.alpha = nn.Parameter(torch.FloatTensor(1).zero_())
    self.beta = nn.Parameter(torch.FloatTensor(1).zero_())

    nn.init.constant_(self.alpha, 1)

  def forward(self, x, res_list):
    res = torch.cat(res_list[1:], dim=1)
    score = self.conv1(res)
    # score = self.conv2(F.leaky_relu(score,0.1))
    score = self.conv2(torch.cat([score, res_list[0]], dim=1))
    i = score
    # score = self.conv3(F.leaky_relu(score,0.1))
    # score = self.conv4(F.leaky_relu(score,0.1))
    # score = self.conv5(F.leaky_relu(score+i,0.1))
    score = self.conv3(torch.tanh(score))
    score = self.conv4(torch.tanh(score))
    score = self.conv5(torch.tanh(score + i))
    # score = self.conv5(score)
    # in
    '''
    t = score
    m = t.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    std = ((t - m) ** 2).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    t = (t - m) / (std.sqrt()) * self.alpha + self.beta
    score = t
    '''
    #
    mask = F.sigmoid(score)

    return mask


class Channel_att_score3(nn.Module):
  def __init__(self, inChannel, num_prev, r=8):
    super(Channel_att_score3, self).__init__()
    self.num_prev = num_prev

    # self.conv = nn.Conv2d(num_prev*inChannel,inChannel,kernel_size=1)
    # self.conv2 = nn.Conv2d(inChannel,inChannel,kernel_size=3,padding=1)
    # self.conv3 = nn.Conv2d(inChannel,1,kernel_size=3,padding=1)
    # self.fc1_1 = nn.Conv2d(1,4,kernel_size=[64,num_prev+1],padding=0,bias=True)
    self.fc1_1 = nn.Conv2d(inChannel, inChannel // r, kernel_size=1,
                           bias=False)
    self.fc1_2 = nn.Conv2d(inChannel // r, inChannel, kernel_size=1,
                           bias=False)
    if self.num_prev > 0:
      self.fc1_3 = nn.Conv2d(inChannel, inChannel // r,
                             kernel_size=[num_prev + 1, 1], padding=0,
                             bias=False)
      self.fc1_4 = nn.Conv2d(inChannel // r, inChannel, kernel_size=1,
                             bias=False)
      # self.fc1_3 = nn.Conv2d(inChannel, inChannel // r, kernel_size=[num_prev, 1], padding=0, bias=True)

    # self.conv4  =nn.Conv2d(2,1,kernel_size=7,padding=3)
    self.alpha = nn.Parameter(torch.FloatTensor(1).zero_())
    self.beta = nn.Parameter(torch.FloatTensor(1).zero_())
    self.register_parameter('norm_alpha', self.alpha)
    self.register_parameter('norm_beta', self.beta)

    self.alpha1 = nn.Parameter(torch.FloatTensor(1).zero_())
    self.beta1 = nn.Parameter(torch.FloatTensor(1).zero_())
    self.register_parameter('norm_alpha1', self.alpha1)
    self.register_parameter('norm_beta1', self.beta1)

    nn.init.constant_(self.alpha, 1)
    nn.init.constant_(self.alpha1, 1)

    self.activ1 = nn.ReLU()
    self.activ2 = nn.Sigmoid()

  def forward(self, x, MP_list, GP_list):

    MP = F.max_pool2d(x, kernel_size=x.size()[2:])
    GP = F.avg_pool2d(x, kernel_size=x.size()[2:])

    MP_list_ = copy.copy(MP_list)
    GP_list_ = copy.copy(GP_list)  # shallow copy
    # MP_list_.append(MP.squeeze(3).squeeze(2))
    # GP_list_.append(GP.squeeze(3).squeeze(2))
    GP_list_.append(GP)

    t2 = GP

    m = t2.mean(dim=1, keepdim=True)
    std = ((t2 - m) ** 2).mean(dim=1, keepdim=True)
    t2 = (t2 - m) / (std.sqrt()) * self.alpha + self.beta

    t2 = self.fc1_1(t2)

    t = (self.fc1_2(self.activ1(t2)))
    output = 1 + torch.tanh(t)

    if self.num_prev == 0:

      # return output,(output)*x
      return output.squeeze(3).squeeze(2), output * x


    else:
      # x2 = torch.cat(GP_list_,dim=2)
      x2 = torch.cat(GP_list_, dim=2)

      m = x2.mean(dim=1, keepdim=True)
      std = ((x2 - m) ** 2).mean(dim=1, keepdim=True)
      x2 = (x2 - m) / (std.sqrt()) * self.alpha1 + self.beta1

      x2 = self.fc1_3(x2)

      # print(x2)
      x2 = self.fc1_4(self.activ1(x2))  # + t
      # print(F.tanh(t))
      output = 1 + torch.tanh(x2)
      # output = (self.activ2(x2))+(self.activ2(t))
      # print(output.size())
      # print(output)
      # output = (output.unsqueeze(2).unsqueeze(3))*x
      return output.squeeze(3).squeeze(2), output * x


class Global_att(nn.Module):
  def __init__(self, inChannel, num_unit, prev):
    super(Global_att, self).__init__()
    self.inChannel = inChannel
    self.num_unit = num_unit
    self.Fuse = nn.Conv2d(inChannel * num_unit, inChannel, 1)
    # self.Fuse = nn.Conv2d(inChannel,inChannel,3,padding=1,groups=num_unit)
    self.Trans = nn.Conv2d(inChannel, inChannel, 3, padding=1)
    # self.Mask = nn.Conv2d(inChannel,1,7,padding=3)
    self.alpha = nn.Parameter(torch.Tensor([1]))
    self.cnt = 0

    nn.init.constant_(self.alpha, 0)

  def forward(self, x_ori, x, res_list, temp_list):
    x_ = torch.cat(res_list, dim=1)
    '''
    x_ = x_.view(-1, self.num_unit, self.inChannel, *x.size()[2:])
    x_ = torch.transpose(x_,1,2).contiguous()
    x_ = x_.view(-1, self.num_unit*self.inChannel, *x.size()[2:])
    '''
    mask = (self.Trans(F.relu(self.Fuse(x_))))
    x = (mask) + x_ori

    # score = self.Mask(F.relu(x))
    # mask = self.Fuse(x_)
    # self.cnt+=1
    # if self.cnt==1600:
    #     print(self.alpha)
    #     self.cnt = 0
    return x


class ST_Unit5(nn.Module):
  def __init__(self, nChannel, prev=0, bias=True, dilation=1, conv1=None,
               conv2=None):  # ch = 128  nch = 64 (split)
    super(ST_Unit5, self).__init__()
    self.nChannel = nChannel
    self.prev = prev

    self.att_c = Channel_att_score3(nChannel, self.prev)
    '''
    self.Conv = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1,bias=bias)
    self.Linear = nn.Conv2d(nChannel, nChannel*9, kernel_size=1, padding=0,bias=bias)
    self.Subpixel = nn.PixelShuffle(3)
    #self.att_s = Spatial_att_score2(prev)

    var = np.sqrt(6 / (64 * 3 * 3 + 64 * 3 * 3))
    self.filters = nn.Parameter(torch.FloatTensor(64,64,3,3).uniform_(-var,var))
    self.register_parameter('conv1x1',self.filters)
    '''
    # self.Spatial_att = Spatial_att()
    # self.conv3x3 = nn.Conv2d(nChannel*(prev+1), nChannel, kernel_size=3, padding=1, bias=bias)
    # self.Conv_1x1 = torch.
    self.Conv_1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3,
                              padding=dilation, dilation=dilation,
                              bias=bias) if conv1 is None else conv1
    self.DWconv_3x3 = nn.Conv2d(nChannel, nChannel, kernel_size=3,
                                padding=1,
                                bias=bias) if conv2 is None else conv2

    # self.Conv_1x1_2 = nn.Conv2d(nChannel*4, nChannel, kernel_size=1, padding=0, bias=bias)
    # self.Channel_att = Channel_att(nChannel,)
    # self.Compress = nn.Conv2d(nChannel*4, nChannel, kernel_size=1, padding=0, bias=bias)
    # self.Fuse = nn.Conv2d(nChannel*2, nChannel, kernel_size=1, padding=0, bias=bias)

    '''
    self.conv3x3_1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)
    self.conv3x3_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=2, dilation=2, bias=bias)
    self.conv1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=3, dilation=3, bias=bias)
    self.conv1x1_2 = nn.Conv2d(nChannel*4, nChannel, kernel_size=1, padding=0, bias=bias)
    '''
    # var = np.sqrt(6 / (64 * 3 * 3 + 64 * 3 * 3))
    # self.filters = nn.Parameter(torch.FloatTensor(64,64,3,3).uniform_(-var,var))
    # self.register_parameter('shared_conv',self.filters)
    # self.conv3x3_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=2, dilation=2,bias=bias)
    # self.conv3x3_3 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=3, dilation=3,bias=bias)
    # var = np.sqrt(6 / (64 * 3 * 3 + 64 * 3 * 3))
    # self.filters = nn.Parameter(torch.FloatTensor(64,64,3,3).uniform_(-var,var))
    # self.register_parameter('shared_conv',self.filters)
    # self.Fuse = nn.Conv2d(nChannel*2, nChannel, kernel_size=1, padding=0,bias=False)
    # if self.prev>0:
    # self.conv1 = nn.Conv2d(nChannel*(self.prev+1), nChannel, kernel_size=1, padding=0, bias=bias)
    # self.conv2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)

  def forward(self, x, pre_res, MP_list, GP_list):
    ori = x[:, -self.nChannel:, :, :]

    scale = None
    '''
    ka = F.avg_pool2d(self.Conv(x),kernel_size=x.size()[2:])
    ka = self.Linear(F.relu(ka))
    ka = self.Subpixel(ka)
    ka_ = ka.view(-1,self.nChannel,9,1)
    ka_ = F.softmax(ka_/8.,dim=2)*9
    ka = ka_.view(-1,self.nChannel,3,3)
    ka = torch.mean(ka,dim=0,keepdim=False).unsqueeze(0)

    x = F.conv2d(x,self.filters*ka,padding=1)
    '''

    x1 = self.DWconv_3x3(F.relu(self.Conv_1x1(x)))  # not use relu
    # x1 = F.relu((self.Conv_1x1(x))) #not use relu
    # scale,x1 = self.Channel_att(x1)
    # scale = F.avg_pool2d(x1,kernel_size=x1.size()[2:])

    i = x1

    # x1 = F.conv2d(F.relu(x1),self.filters,bias=None,padding=2,dilation=2)
    # x1 = F.conv2d(F.relu(x1),self.filters,bias=None,padding=3,dilation=3)
    # x1 = self.Spatial_att(x1)
    # x1 = self.Spatial_conv2(F.relu(x1))
    # x1 = self.Spatial_att(x1)
    # x1 = self.att_c(x1, pre_res)

    # x1 = self.Fuse(torch.cat([i,x1],dim=1)) #not use relu
    # x1 = self.att_c(x1,pre_res)
    # x1 = ((self.conv3x3_2(F.relu(self.conv3x3_1(x)))))
    # x1 = F.conv2d(x,self.filters,bias=None,padding=1,dilation=1)
    # x1 = (F.conv2d(F.relu(x1),self.filters,bias=None,padding=2,dilation=2))
    # x1 = (F.conv2d(F.relu(x1),self.filters,bias=None,padding=3,dilation=3))

    # x1 = ((self.conv1x1(F.relu(x1)))) # self.conv1x1(x1)#
    # x1 = F.conv2d(x,self.filters,bias=None,padding=1)
    # x1 = F.conv2d(F.relu(x1),self.filters,bias=None,padding=2,dilation=2)

    # x1 = self.conv3x3_3(F.relu(self.conv3x3_2(x1)))

    scale, x1 = self.att_c(x1, MP_list, GP_list)
    # print(len(MP_list))
    #
    # x1 = self.att_s(x1, pre_res)

    # x1 = x1+i
    # if not self.prev:
    #    return x1,x1 + ori
    # else:
    # x1 = self.Fuse(torch.cat([i,x1],dim=1))
    # score = F.tanh(self.conv2(F.leaky_relu(self.conv1(torch.cat(pre_list,dim=1)),0.1)))
    # bias = F.tanh(self.conv4(torch.cat([F.leaky_relu(self.conv3(torch.cat(pre_res,dim=1)),0.1),F.leaky_relu(x1,0.1)],dim=1)))
    # bias = (self.conv4(torch.cat([F.leaky_relu(self.conv3(torch.cat(pre_res,dim=1)),0.1),F.relu(x1)],dim=1)))
    # x1 = (score)*x1
    return x1, x1 + ori, scale


class ST_Block5(nn.Module):
  def __init__(self, num_unit, nChannel, prev=0,
               bias=True):  # ch = 128  nch = 64 (split)
    super(ST_Block5, self).__init__()
    self.nChannel = nChannel
    self.num_unit = num_unit
    self.Conv_1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1,
                              dilation=1, bias=bias)
    self.DWconv_3x3 = nn.Conv2d(nChannel, nChannel, kernel_size=3,
                                padding=1, bias=bias)
    self.prev = prev
    # self.att_s = Spatial_att_score3(nChannel,prev)#prev)
    # self.att_c = Channel_att_score3(nChannel, self.prev, nChannel)
    # self.Conv1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)
    # self.att_c = Channel_att_score2(nChannel,self.num_unit,nChannel)

    self.Spatial_att = Global_att(nChannel, num_unit, prev)
    self.unit = torch.nn.ModuleList(
      [ST_Unit5(64, i, dilation=1, conv1=None, conv2=None) for i in
       range(num_unit)])
    self.Fuse = nn.Conv2d(nChannel * 2, nChannel, kernel_size=1, padding=0,
                          bias=bias)  # *(num_unit+1)
    self.Infuse = nn.Conv2d(nChannel * (prev + 1), nChannel, kernel_size=1,
                            padding=0, bias=bias)
    if prev > 0: self.Conv = nn.Conv2d(nChannel * (2), nChannel,
                                       kernel_size=1, padding=0, bias=bias)
    # self.lastFuse = nn.Conv2d(nChannel * (prev + 1), nChannel, kernel_size=1, padding=0, bias=bias)
    # self.Mask1 = nn.Conv2d(nChannel*(num_unit), nChannel, kernel_size=1, padding=0, bias=bias)
    # self.Mask2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)

  def forward(self, x, res_list, temp_list):

    i = x[:, -self.nChannel:, :, :]
    x = i

    if self.prev == 0:
      x = self.Infuse(x)
    else:
      r = torch.cat(res_list, dim=1)
      # x = torch.cat([x, r], dim=1)
      x = self.Infuse(r)

    ori = x
    # x = self.Conv1(x)#fore or after?
    out_list = []
    x_list = []
    MP_list = []
    GP_list = []
    for i, model in enumerate(self.unit):
      res, x, scale = model(x, out_list, MP_list, GP_list)
      if i != self.num_unit - 1:  # if False:
        # pass
        # MP_list.append(F.max_pool2d(res, kernel_size=x.size()[2:]))
        GP_list.append(F.avg_pool2d(res, kernel_size=x.size()[2:]))
      out_list.append(res)
      x_list.append(x)
    # x = self.Spatial_att(x,out_list)
    # res_x = torch.cat(out_list,dim=1)
    # mask = F.sigmoid(self.Mask2(F.relu(self.Mask1(res_x))))

    # x = self.Fuse(x)
    # x = self.att_c(x,res_list)
    # x = self.Channel_att(x)
    x_ = self.Spatial_att(ori, x, out_list, temp_list)
    x = self.Fuse(torch.cat([x_, x], dim=1))
    # if self.prev > 0:
    #            x = self.Conv(torch.cat([x, temp_list[-1]], dim=1))
    '''
    if self.prev==0:x = self.lastFuse(x)
    else:
        r = torch.cat(temp_list,dim=1)
        x = torch.cat([x,r],dim=1)
        x = self.lastFuse(x)
    '''
    return x, x + i, x_list  # x+i
