from VSR.Util.Utility import Vgg
from PIL import Image
import tensorflow as tf
import numpy as np

URL = 'data/set5_x2/img_001_SRF_2_LR.png'
image_boy = np.asarray(Image.open(URL))


def test_vgg16():
    vgg = Vgg(False, vgg=Vgg.VGG16)
    x = np.random.normal(size=[16, 128, 128, 3])
    y = vgg(x)
    assert y.shape == (16,)
    with tf.Session() as sess:
        sess.run(y)


def test_vgg19():
    vgg = Vgg(False, vgg=Vgg.VGG19)
    x = np.random.normal(size=[16, 128, 128, 3])
    y = vgg(x, 'block2_conv2')
    assert y.shape == (16, 64, 64, 128)
    with tf.Session() as sess:
        sess.run(y)


def test_vgg_classify():
    vgg16 = Vgg(True, vgg=Vgg.VGG16)
    vgg19 = Vgg(True, vgg=Vgg.VGG19)
    x = np.expand_dims(image_boy, 0)
    y1 = vgg16(x)
    y2 = vgg19(x)
    with tf.Session() as sess:
        y1, y2 = sess.run([y1, y2])
        assert y2[0].tolist().index(y2.max()) == y1[0].tolist().index(y1.max())


def test_multiple_call():
    vgg1 = Vgg(False, vgg=Vgg.VGG16)
    vgg2 = Vgg(False, vgg=Vgg.VGG16)
    x = np.expand_dims(image_boy, 0)
    y1 = vgg1(x)
    y2 = vgg2(x)
    y3 = vgg2(x.copy())
    with tf.Session() as sess:
        sess.run([y1, y2, y3])


if __name__ == '__main__':
    test_multiple_call()
    pass
