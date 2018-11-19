from VSR.Util.ImageProcess import imread, rgb_to_yuv, array_to_img
from VSR.Util import Utility as U
import tensorflow as tf

tf.enable_eager_execution()
import numpy as np
from PIL import Image

URL = 'data/set5_x2/img_001_SRF_2_LR.png'


def test_rgb2yuv():
    img = imread(URL)
    img = img.astype('float32')

    yuv = rgb_to_yuv(img, 255, 'matlab')
    array_to_img(yuv).show()


def test_resize_2x2():
    scale = 2
    img = np.array([[1, 2], [3, 4]], np.uint8)
    imgp = Image.fromarray(img, 'L')
    img = np.reshape(img, [1, 2, 2, 1])
    y = U.upsample(img.astype(np.float32), scale)
    y = y.numpy()[0, ..., 0].astype('uint8')
    imgp = imgp.resize([2 * scale, 2 * scale], Image.BICUBIC)
    yp = np.array(imgp)
    yf = tf.image.resize_bicubic(img, [2 * scale, 2 * scale])
    yf = yf.numpy()[0, ..., 0].astype('uint8')
    diff = yp - y


def test_resize_img():
    scale = 2
    img = Image.open(URL)
    w = img.width
    h = img.height
    imgp = img.resize([w * scale, h * scale], Image.BICUBIC)
    yp = np.array(imgp)
    img = np.resize(img, [1, h, w, 3])
    y = U.upsample(img.astype(np.float32), scale)
    y = y.numpy()[0].astype('uint8')
    yf = tf.image.resize_bicubic(img, [h * scale, w * scale])
    yf = yf.numpy()[0].astype('uint8')
    diff = yp - y


if __name__ == '__main__':
    test_resize_2x2()
    test_resize_img()
    pass
