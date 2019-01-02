from PIL import Image
import tensorflow as tf
from VSR.Framework.Noise import *


CRF = np.load('../Data/crf.npz')
URL = "data/set5_x2/img_001_SRF_2_LR.png"


def add_noise_crf(image, index=0):
    image = image.astype('float') / 255
    icrf = CRF['icrf'][index]
    crf = CRF['crf'][index]
    irr = camera_response_function(image, icrf)
    noise = gaussian_poisson_noise(irr)
    noisy = camera_response_function(irr + noise, crf) * 255
    noisy = Image.fromarray(noisy.clip(0, 255).astype('uint8'), 'RGB')
    noisy.show()


def tf_add_noise_crf(image, index=0):
    image = image.astype('float') / 255
    icrf = CRF['icrf'][index]
    crf = CRF['crf'][index]
    image = tf.convert_to_tensor([image.tolist()])
    icrf = tf.convert_to_tensor(icrf)
    crf = tf.convert_to_tensor(crf)
    irr = tf_camera_response_function(image, icrf)
    noise = tf_gaussian_poisson_noise(irr)
    noisy = tf_camera_response_function(irr + noise, crf) * 255
    if tf.executing_eagerly():
        noisy = noisy.numpy()
    else:
        with tf.Session():
            noisy = noisy.eval()
    noisy = Image.fromarray(noisy[0].clip(0, 255).astype('uint8'), 'RGB')
    noisy.show()


if __name__ == '__main__':
    img = Image.open(URL)
    add_noise_crf(np.array(img), 15)
    tf_add_noise_crf(np.array(img), 16)
