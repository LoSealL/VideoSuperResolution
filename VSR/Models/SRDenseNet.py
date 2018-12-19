"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 19th 2018

Image Super-Resolution Using Dense Skip Connections
See http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
"""


from ..Framework.SuperResolution import SuperResolution
from ..Arch import Dense

import tensorflow as tf


def _denormalize(inputs):
    return (inputs + 0) * 255


def _normalize(inputs):
    return inputs / 255


class SRDenseNet(SuperResolution):
    """Image Super-Resolution Using Dense Skip Connections.
    Args:
        n_blocks: number of dense blocks.
    """
    def __init__(self, name='srdensenet', n_blocks=8, **kwargs):
        super(SRDenseNet, self).__init__(**kwargs)
        self.name = name
        self.n_blocks = n_blocks

    def build_graph(self):
        super(SRDenseNet, self).build_graph()
        with tf.variable_scope(self.name):
            inputs_norm = _normalize(self.inputs_preproc[-1])
            feat = [self.conv2d(inputs_norm, 64, 3)]
            for i in range(self.n_blocks):
                feat.append(Dense.dense_block(self, feat[-1]))
            bottleneck = self.conv2d(tf.concat(feat, -1), 256, 1)
            sr = self.upscale(bottleneck, 'deconv', direct_output=False)
            sr = self.conv2d(sr, self.channel, 3)
            self.outputs.append(_denormalize(sr))

    def build_loss(self):
        mse, loss = super(SRDenseNet, self).build_loss()
        self.train_metric['mse'] = mse
        self.train_metric['loss'] = loss
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(
            self.label[-1], self.outputs[-1], max_val=255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(
            self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        super(SRDenseNet, self).build_summary()
        tf.summary.image('sr', self.outputs[-1])
