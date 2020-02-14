import math
import torch


class DenseBlock(torch.nn.Module):
  def __init__(self, input_size, output_size, bias=True, activation='relu',
               norm='batch'):
    super(DenseBlock, self).__init__()
    self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

    self.norm = norm
    if self.norm == 'batch':
      self.bn = torch.nn.BatchNorm1d(output_size)
    elif self.norm == 'instance':
      self.bn = torch.nn.InstanceNorm1d(output_size)

    self.activation = activation
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    if self.norm is not None:
      out = self.bn(self.fc(x))
    else:
      out = self.fc(x)

    if self.activation is not None:
      return self.act(out)
    else:
      return out


class ConvBlock(torch.nn.Module):
  def __init__(self, input_size, output_size, kernel_size=3, stride=1,
               padding=1, bias=True, activation='prelu', norm=None):
    super(ConvBlock, self).__init__()
    self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride,
                                padding, bias=bias)

    self.norm = norm
    if self.norm == 'batch':
      self.bn = torch.nn.BatchNorm2d(output_size)
    elif self.norm == 'instance':
      self.bn = torch.nn.InstanceNorm2d(output_size)

    self.activation = activation
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    if self.norm is not None:
      out = self.bn(self.conv(x))
    else:
      out = self.conv(x)

    if self.activation is not None:
      return self.act(out)
    else:
      return out


class DeconvBlock(torch.nn.Module):
  def __init__(self, input_size, output_size, kernel_size=4, stride=2,
               padding=1, bias=True, activation='prelu', norm=None):
    super(DeconvBlock, self).__init__()
    self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size,
                                           stride, padding, bias=bias)

    self.norm = norm
    if self.norm == 'batch':
      self.bn = torch.nn.BatchNorm2d(output_size)
    elif self.norm == 'instance':
      self.bn = torch.nn.InstanceNorm2d(output_size)

    self.activation = activation
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    if self.norm is not None:
      out = self.bn(self.deconv(x))
    else:
      out = self.deconv(x)

    if self.activation is not None:
      return self.act(out)
    else:
      return out


class ResnetBlock(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True,
               activation='prelu', norm='batch'):
    super(ResnetBlock, self).__init__()
    self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride,
                                 padding, bias=bias)
    self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride,
                                 padding, bias=bias)

    self.norm = norm
    if self.norm == 'batch':
      self.bn = torch.nn.BatchNorm2d(num_filter)
    elif norm == 'instance':
      self.bn = torch.nn.InstanceNorm2d(num_filter)

    self.activation = activation
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    residual = x
    if self.norm is not None:
      out = self.bn(self.conv1(x))
    else:
      out = self.conv1(x)

    if self.activation is not None:
      out = self.act(out)

    if self.norm is not None:
      out = self.bn(self.conv2(out))
    else:
      out = self.conv2(out)

    out = torch.add(out, residual)

    if self.activation is not None:
      out = self.act(out)

    return out


class UpBlock(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True,
               activation='prelu', norm=None):
    super(UpBlock, self).__init__()
    self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                              padding, activation, norm=None)
    self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    h0 = self.up_conv1(x)
    l0 = self.up_conv2(h0)
    h1 = self.up_conv3(l0 - x)
    return h1 + h0


class UpBlockPix(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,
               bias=True, activation='prelu', norm=None):
    super(UpBlockPix, self).__init__()
    self.up_conv1 = Upsampler(scale, num_filter)
    self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                              padding, activation, norm=None)
    self.up_conv3 = Upsampler(scale, num_filter)

  def forward(self, x):
    h0 = self.up_conv1(x)
    l0 = self.up_conv2(h0)
    h1 = self.up_conv3(l0 - x)
    return h1 + h0


class D_UpBlock(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
               num_stages=1, bias=True, activation='prelu', norm=None):
    super(D_UpBlock, self).__init__()
    self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
                          activation, norm=None)
    self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                              padding, activation, norm=None)
    self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    x = self.conv(x)
    h0 = self.up_conv1(x)
    l0 = self.up_conv2(h0)
    h1 = self.up_conv3(l0 - x)
    return h1 + h0


class D_UpBlockPix(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
               num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
    super(D_UpBlockPix, self).__init__()
    self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
                          activation, norm=None)
    self.up_conv1 = Upsampler(scale, num_filter)
    self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                              padding, activation, norm=None)
    self.up_conv3 = Upsampler(scale, num_filter)

  def forward(self, x):
    x = self.conv(x)
    h0 = self.up_conv1(x)
    l0 = self.up_conv2(h0)
    h1 = self.up_conv3(l0 - x)
    return h1 + h0


class DownBlock(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True,
               activation='prelu', norm=None):
    super(DownBlock, self).__init__()
    self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                  padding, activation, norm=None)
    self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    l0 = self.down_conv1(x)
    h0 = self.down_conv2(l0)
    l1 = self.down_conv3(h0 - x)
    return l1 + l0


class DownBlockPix(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,
               bias=True, activation='prelu', norm=None):
    super(DownBlockPix, self).__init__()
    self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.down_conv2 = Upsampler(scale, num_filter)
    self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    l0 = self.down_conv1(x)
    h0 = self.down_conv2(l0)
    l1 = self.down_conv3(h0 - x)
    return l1 + l0


class D_DownBlock(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
               num_stages=1, bias=True, activation='prelu', norm=None):
    super(D_DownBlock, self).__init__()
    self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
                          activation, norm=None)
    self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride,
                                  padding, activation, norm=None)
    self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    x = self.conv(x)
    l0 = self.down_conv1(x)
    h0 = self.down_conv2(l0)
    l1 = self.down_conv3(h0 - x)
    return l1 + l0


class D_DownBlockPix(torch.nn.Module):
  def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
               num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
    super(D_DownBlockPix, self).__init__()
    self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
                          activation, norm=None)
    self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)
    self.down_conv2 = Upsampler(scale, num_filter)
    self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                padding, activation, norm=None)

  def forward(self, x):
    x = self.conv(x)
    l0 = self.down_conv1(x)
    h0 = self.down_conv2(l0)
    l1 = self.down_conv3(h0 - x)
    return l1 + l0


class PSBlock(torch.nn.Module):
  def __init__(self, input_size, output_size, scale_factor, kernel_size=3,
               stride=1, padding=1, bias=True, activation='prelu',
               norm='batch'):
    super(PSBlock, self).__init__()
    self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor ** 2,
                                kernel_size, stride, padding, bias=bias)
    self.ps = torch.nn.PixelShuffle(scale_factor)

    self.norm = norm
    if self.norm == 'batch':
      self.bn = torch.nn.BatchNorm2d(output_size)
    elif norm == 'instance':
      self.bn = torch.nn.InstanceNorm2d(output_size)

    self.activation = activation
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    if self.norm is not None:
      out = self.bn(self.ps(self.conv(x)))
    else:
      out = self.ps(self.conv(x))

    if self.activation is not None:
      out = self.act(out)
    return out


class Upsampler(torch.nn.Module):
  def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
    super(Upsampler, self).__init__()
    modules = []
    for _ in range(int(math.log(scale, 2))):
      modules.append(
        ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None,
                  norm=None))
      modules.append(torch.nn.PixelShuffle(2))
      if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
      # modules.append(torch.nn.PReLU())
    self.up = torch.nn.Sequential(*modules)

    self.activation = act
    if self.activation == 'relu':
      self.act = torch.nn.ReLU(True)
    elif self.activation == 'prelu':
      self.act = torch.nn.PReLU()
    elif self.activation == 'lrelu':
      self.act = torch.nn.LeakyReLU(0.2, True)
    elif self.activation == 'tanh':
      self.act = torch.nn.Tanh()
    elif self.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid()

  def forward(self, x):
    out = self.up(x)
    if self.activation is not None:
      out = self.act(out)
    return out


class Upsample2xBlock(torch.nn.Module):
  def __init__(self, input_size, output_size, bias=True, upsample='deconv',
               activation='relu', norm='batch'):
    super(Upsample2xBlock, self).__init__()
    scale_factor = 2
    # 1. Deconvolution (Transposed convolution)
    if upsample == 'deconv':
      self.upsample = DeconvBlock(input_size, output_size,
                                  kernel_size=4, stride=2, padding=1,
                                  bias=bias, activation=activation, norm=norm)

    # 2. Sub-pixel convolution (Pixel shuffler)
    elif upsample == 'ps':
      self.upsample = PSBlock(input_size, output_size,
                              scale_factor=scale_factor,
                              bias=bias, activation=activation, norm=norm)

    # 3. Resize and Convolution
    elif upsample == 'rnc':
      self.upsample = torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        ConvBlock(input_size, output_size,
                  kernel_size=3, stride=1, padding=1,
                  bias=bias, activation=activation, norm=norm)
      )

  def forward(self, x):
    out = self.upsample(x)
    return out
