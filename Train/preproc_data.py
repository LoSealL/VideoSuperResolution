"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug. 16th 2018

pre-process images in dataset
- cal mean and covariance
- ~random crop~
"""

import numpy as np
import tensorflow as tf

from VSR.DataLoader.Dataset import load_datasets
from VSR.DataLoader.Loader import QuickLoader, BasicLoader
from VSR.Util import FID
from VSR.Util.ImageProcess import imresize, array_to_img

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

FLAGS = tf.flags.FLAGS
SAVE = {}


def main(*args):
    for name in FLAGS.dataset:
        name = name.upper()
        d = DATASETS.get(name)
        if not d:
            tf.logging.error('Could not find ' + name)
            return

        # calc mean [R G B]
        loader = QuickLoader(1, d, 'train', 1, convert_to='RGB')
        colors = []
        for img, _, _ in loader.make_one_shot_iterator(shard=8):
            rgb = np.reshape(img, [-1, 3])
            colors.append(rgb)
        colors = np.concatenate(colors)
        mean_colors = colors.mean(axis=0, keepdims=True)
        SAVE[f'{name}_MEAN'] = mean_colors
        if FLAGS.std:
            std_colors = colors.std(axis=0, keepdims=True)
            SAVE[f'{name}_STD'] = std_colors

        if FLAGS.fid:
            # activation of pool 3
            inception_pb = FID.check_or_download_inception(FLAGS.model_path)
            FID.create_inception_graph(inception_pb)
            imgs = []
            for img, _, _ in loader.make_one_shot_iterator(shard=8):
                imgs += [imresize(array_to_img(img[0], 'RGB'), 0, size=[299, 299])]
            imgs = np.stack(imgs)

            with tf.Session() as sess:
                acts = FID.get_activations(imgs, sess)
                mu = acts.mean(axis=0)
                sigma = np.cov(acts, rowvar=False)
            SAVE[f'{name}_FID_MU'] = mu
            SAVE[f'{name}_FID_SIGMA'] = sigma

    np.savez_compressed(FLAGS.output, **SAVE)


if __name__ == '__main__':
    tf.flags.DEFINE_list('dataset', None, 'a string or a list of strings, names of datasets')
    tf.flags.DEFINE_string('output', 'data.npz', 'name of the output file')
    tf.flags.DEFINE_bool('std', False, 'calc stddev of image colors (require large memory)')
    tf.flags.DEFINE_bool('fid', False, 'calc mean and cov for pool_3 layer of the inception model')
    tf.flags.DEFINE_string('model_path', None, 'path to inception_v1 model')
    tf.app.run(main)
