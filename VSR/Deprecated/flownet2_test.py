#! python3
# flownet2-tf from sampepose, forked at
# https://github.com/loseall/flownet2-tf

# gcc-5 is required to build so files

# Date: Oct. 31st 2018
# Author: Wenyi Tang
# Email: wenyi.tang@intel.com

# pre-trained FlowNet2 forked from sampepose/flownet2-tf
# See https://github.com/LoSealL/flownet2-tf
from src.flownet2.flownet2 import FlowNet2 as Net
from src.flow_warp import flow_warp
import tensorflow as tf
import numpy as np

from PIL import Image
from VSR.Framework.Callbacks import _viz_flow
from VSR.Framework.Motion import warp

checkpoint = '/home/intel/works/flownet2-tf/checkpoints/FlowNet2/flownet-2.ckpt-0'
IMG1 = 'data/flying_chair/img0.png'
IMG2 = 'data/flying_chair/img1.png'


def test_restore_checkpoint():
    """restore from checkpoint"""

    net = Net(2)  # Test Mode
    with tf.Session() as sess:
        net.restore(sess, checkpoint)


def test_output_flow():
    """calc flow"""

    net = Net(2)
    imga = Image.open(IMG1)
    imgb = Image.open(IMG2)
    imga = np.asarray(imga).astype('float32') / 255
    imgb = np.asarray(imgb).astype('float32') / 255
    imga = imga[..., [2, 1, 0]]
    imgb = imgb[..., [2, 1, 0]]

    with tf.Session() as sess:
        net.restore(sess, checkpoint)
        imga = np.expand_dims(imga, 0)
        imgb = np.expand_dims(imgb, 0)

        print(imga.shape, imgb.shape)

        flow = net.flow_results.eval(feed_dict={
            'flow/inputa:0': imga,
            'flow/inputb:0': imgb,
        })
        x = flow[0]

    x = _viz_flow(x[..., 0], x[..., 1])
    Image.fromarray(x, 'RGB').save('flow.png')


def test_flow_warp():
    net = Net(2)
    imga = Image.open(IMG1)
    imgb = Image.open(IMG2)
    imga = np.asarray(imga).astype('float32') / 255
    imgb = np.asarray(imgb).astype('float32') / 255
    imga = imga[..., [2, 1, 0]]
    imgb = imgb[..., [2, 1, 0]]

    with tf.Session() as sess:
        net.restore(sess, checkpoint)
        imga = np.expand_dims(imga, 0)
        imgb = np.expand_dims(imgb, 0)

        print(imga.shape, imgb.shape)

        flow = net.flow_results.eval(feed_dict={
            'flow/inputa:0': imga,
            'flow/inputb:0': imgb,
        })
        x = flow

        u = x[..., 0]
        v = x[..., 1]

        image = tf.constant(imgb[..., [2, 1, 0]])
        wp = warp(image, u, v, additive_warp=True).eval()[0]
        wp = (wp * 255).astype('uint8')
        Image.fromarray(wp, 'RGB').show()

        wp = flow_warp(image, flow).eval()[0]
        wp = (wp * 255).astype('uint8')
        Image.fromarray(wp, 'RGB').show()


def test_reuse():
    net = Net(2)
    x1 = tf.ones([1, 128, 128, 3])
    x2 = tf.ones([1, 128, 128, 3])
    y1 = tf.ones([1, 128, 128, 3])
    y2 = tf.ones([1, 128, 128, 3])

    f1 = net(x1, x2)
    f2 = net(y1, y2)
    with tf.Session() as sess:
        net.restore2(sess, checkpoint, 'FlowNet2')
        f1 = f1.eval()
        f2 = f2.eval()
        assert np.all(np.abs(f1 - f2) < 1.e-6)


if __name__ == '__main__':
    # test_restore_checkpoint()
    # test_output_flow()
    # test_flow_warp()
    test_reuse()
    pass
