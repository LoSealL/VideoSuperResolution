"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

Common optimizer framework for single-image SR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...Framework.Summary import get_writer
from ...Util import Metrics
from ...Util.Utility import pad_if_divide
from ..Model import SuperResolution
from ..Ops.Blocks import HighPassFilter, LowPassFilter
from ..Ops.Discriminator import PatchGAN
from ..Ops.Loss import DiscriminatorLoss, GeneratorLoss, VggFeatureLoss


def get_opt(opt_config, params, learning_rate):
    """Get optimizer functor
    """

    opt = torch.optim.Adam(params, lr=learning_rate)
    if opt_config is None:
        pass
    elif opt_config.get('name') == 'Adadelta':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.Adadelta(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'Adagrad':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.Adagrad(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'Adam':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.Adam(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'SparseAdam':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.SparseAdam(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'Adamax':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.Adamax(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'ASGD':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.ASGD(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'SGD':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.SGD(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'LBFGS':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.LBFGS(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'Rprop':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.Rprop(params, lr=learning_rate, **kwargs)
    elif opt_config.get('name') == 'RMSprop':
        kwargs = opt_config
        kwargs.pop('name')
        opt = torch.optim.RMSprop(params, lr=learning_rate, **kwargs)
    return opt


def get_pix_cri(cri_config=None):
    """Get pixel/image loss function

    Args:
        cri_config: a dict with `name` and other config keys
    """

    if cri_config is None:
        return nn.L1Loss()
    cri_name = cri_config.get('name', '').upper()
    if cri_name in ('L1', 'MAE'):
        return nn.L1Loss()
    if cri_name in ('L2', 'MSE'):
        return nn.MSELoss()
    if cri_name in ('SMOOTH', 'L1L2'):
        return nn.SmoothL1Loss()
    return nn.L1Loss()


class L1Optimizer(SuperResolution):
    """Simple L1 pixel optimizer framework

    Optimize image with simple L1 loss function.
    """

    def __init__(self, scale=1, channel=3, **kwargs):
        super(L1Optimizer, self).__init__(scale, channel)
        # gradient clip
        self.clip = kwargs.get('clip')
        # default use Adam with beta1=0.9 and beta2=0.999
        self.opt = get_opt(kwargs.get('opt'), self.trainable_variables(), 1e-4)
        self.padding = kwargs.get('padding', 0)

    def fn(self, x):
        """An abstract function used to inference network
        """
        raise NotImplementedError

    def train(self, inputs, labels, learning_rate=None):
        sr = self.fn(inputs[0])
        loss = F.l1_loss(sr, labels[0])
        if learning_rate:
            for param_group in self.opt.param_groups:
                param_group["lr"] = learning_rate
        self.opt.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(
                self.trainable_variables(), self.clip)
        self.opt.step()
        return {'l1': loss.detach().cpu().numpy()}

    def eval(self, inputs, labels=None, **kwargs):
        metrics = {}
        _lr = inputs[0]
        if self.padding:
            lr = pad_if_divide(_lr, self.padding)
            a = lr.size(2) - _lr.size(2)
            b = lr.size(3) - _lr.size(3)
            slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
            slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
            sr = self.fn(lr)[..., slice_h, slice_w]
        else:
            sr = self.fn(_lr)
        sr = sr.cpu().detach()
        if labels is not None:
            metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
            writer = get_writer(self.name)
            if writer is not None:
                step = kwargs.get('epoch')
                writer.image('sr', sr.clamp(0, 1), max=1, step=step)
                writer.image('gt', labels[0], max=1, step=step)
        return [sr.numpy()], metrics

    def export(self, export_dir):
        """An example of how to export ONNX format"""

        # ONNX needs input placeholder to export model!
        # Sounds stupid to set a 48x48 inputs.

        name, model = self.modules.popitem(last=False)
        device = list(model.parameters())[0].device
        inputs = torch.randn(1, self.channel, 48, 48, device=device)
        torch.onnx.export(model, (inputs,), export_dir / f'{name}.onnx')


class PerceptualOptimizer(L1Optimizer):
    """Predefined optimizer framework for SISR task in name of `SRGAN` manner

    Implement `fn` function in subclass
    """

    def __init__(self, scale, channel, image_weight=1, feature_weight=0,
                 gan_weight=0, **kwargs):
        super(PerceptualOptimizer, self).__init__(scale, channel, **kwargs)
        self.use_vgg = feature_weight > 0
        self.use_gan = gan_weight > 0
        if self.use_vgg:
            # tricks: do not save weights of vgg
            feature_lists = kwargs.get('vgg_features', ['block5_conv4'])
            self.feature = [VggFeatureLoss(feature_lists, True)]
        if self.use_gan:
            # define D-net
            dnet = kwargs.get('discriminator', PatchGAN)
            dnet_kw = kwargs.get('discriminator_kwargs', {
                'channel': channel,
                'num_layers': 3,
                'norm': 'BN',
                'activation': 'leaky',
                'negative_slope': 0.2,
            })
            self.dnet = dnet(**dnet_kw)
            self.optd = torch.optim.Adam(
                self.trainable_variables('dnet'), 1e-4)
        self.use_filter = kwargs.get('gan_with_filter')
        if self.use_filter:
            config = kwargs.get('filter_kwargs', {})
            self.high_filter = HighPassFilter(**config)
            self.low_filter = LowPassFilter(**config)
        # image, vgg, gan
        self.w = [image_weight, feature_weight, gan_weight]
        self.pixel_cri = get_pix_cri(kwargs.get('cri_image'))
        self.gen_cri = GeneratorLoss(kwargs.get('cri_gan', 'GAN'))
        self.disc_cri = DiscriminatorLoss(kwargs.get('cri_gan', 'GAN'))

    def cuda(self):
        super(PerceptualOptimizer, self).cuda()
        if self.use_vgg > 0:
            self.feature[0].cuda()

    def train(self, inputs, labels, learning_rate=None):
        sr = self.fn(inputs[0])
        for opt in self.opts.values():
            if learning_rate:
                for param_group in opt.param_groups:
                    param_group["lr"] = learning_rate
        if self.use_filter:
            image_loss = self.pixel_cri(
                self.low_filter(sr), self.low_filter(labels[0]))
        else:
            image_loss = self.pixel_cri(sr, labels[0])
        loss = image_loss * self.w[0]
        log = {
            'image_loss': image_loss.detach().cpu().numpy()
        }
        if self.use_vgg:
            self.feature[0].eval()
            feat_fake = self.feature[0](sr)[0]
            feat_real = self.feature[0](labels[0])[0].detach()
            feature_loss = self.pixel_cri(feat_fake, feat_real)
            loss += feature_loss * self.w[1]
            log.update(feature=feature_loss.detach().cpu().numpy())
        if self.use_gan:
            for p in self.dnet.parameters():
                p.requires_grad = False
            if self.use_filter:
                fake = self.dnet(self.high_filter(sr))
                real = self.dnet(self.high_filter(labels[0])).detach()
            else:
                fake = self.dnet(sr)
                real = self.dnet(labels[0]).detach()
            gen_loss = self.gen_cri(fake, real)
            loss += gen_loss * self.w[2]
            log.update(gen=gen_loss.detach().cpu().numpy())
        # update G
        self.opt.zero_grad()
        loss.backward()
        if self.clip:
            clip = self.clip / learning_rate
            torch.nn.utils.clip_grad_norm_(self.trainable_variables(), clip)
        self.opt.step()
        if self.use_gan:
            # update D
            for p in self.dnet.parameters():
                p.requires_grad = True
            if self.use_filter:
                disc_fake = self.dnet(self.high_filter(sr.detach()))
                disc_real = self.dnet(self.high_filter(labels[0]))
            else:
                disc_fake = self.dnet(sr.detach())
                disc_real = self.dnet(labels[0])
            disc_loss = self.disc_cri(disc_fake, disc_real)
            self.optd.zero_grad()
            disc_loss.backward()
            self.optd.step()
            log.update(disc=disc_loss.detach().cpu().numpy())
        return log
