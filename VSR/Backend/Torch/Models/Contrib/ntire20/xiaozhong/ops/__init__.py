#  Copyright (c) 2017-2020 Apache 2.0.
#  Author: Xiaozhong Ji
#  Update: 2020 - 5 - 28

from .discriminator import (
  Discriminator_VGG_128, Discriminator_VGG_256, Discriminator_VGG_512,
  NLayerDiscriminator, VGGFeatureExtractor
)
from .network import RRDBNet


####################
# define network
####################

def define_G(which_model='RRDBNet', **opt):
  """
  Generator
  :param which_model:
  :param opt:
  :return:
  """

  if which_model == 'RRDBNet':
    return RRDBNet(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'],
                   nb=opt['nb'])
  else:
    raise NotImplementedError(f'Generator model [{which_model}] not recognized')


def define_D(which_model='NLayerDiscriminator', **opt):
  """
  Discriminator
  :param which_model:
  :param opt:
  :return:
  """

  if which_model == 'discriminator_vgg_128':
    netD = Discriminator_VGG_128(in_nc=opt['in_nc'], nf=opt['nf'])
  elif which_model == 'discriminator_vgg_256':
    netD = Discriminator_VGG_256(in_nc=opt['in_nc'], nf=opt['nf'])
  elif which_model == 'discriminator_vgg_512':
    netD = Discriminator_VGG_512(in_nc=opt['in_nc'], nf=opt['nf'])
  elif which_model == 'NLayerDiscriminator':
    netD = NLayerDiscriminator(input_nc=opt['in_nc'], ndf=opt['nf'],
                               n_layers=opt['nlayer'])
  else:
    raise NotImplementedError(
        f'Discriminator model [{which_model}] not recognized')
  return netD


def define_F(use_bn=False):
  """
  Define Network used for Perceptual Loss
  PyTorch pre-trained VGG19-54, before ReLU.
  :param use_bn:
  :return:
  """

  feature_layer = 49 if use_bn else 34
  netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                             use_input_norm=True)
  netF.eval()  # No need to train
  return netF
