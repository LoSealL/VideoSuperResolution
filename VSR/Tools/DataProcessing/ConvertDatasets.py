"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan 7th, 2019

Convert datasets into TFRecords
Usage:
  ConvertDatasets --dataset=div2k --data_config=Data/datasets.yaml --save_dir=.
"""

import tensorflow as tf
import numpy as np
import io
import tqdm
from pathlib import Path
from PIL import Image

from VSR.DataLoader.Loader import QuickLoader
from VSR.Util.Config import Config
from VSR.Tools import Run
from VSR.Tools.DataProcessing.Util import make_tensor_label_records
from VSR.Framework import Noise

# TODO: Add docs (2019-1-8)
tf.flags.DEFINE_string('method', 'train', help="")
tf.flags.DEFINE_boolean('augment', False, help="")
tf.flags.DEFINE_integer('scale', 4, help="")
tf.flags.DEFINE_integer('channel', 3, help="")
tf.flags.DEFINE_integer('patch_size', 48, help="")
tf.flags.DEFINE_integer('batch', 1, help="")
tf.flags.DEFINE_integer('depth', 1, help="")
tf.flags.DEFINE_integer('num', 10000, help="")
tf.flags.DEFINE_integer('jpeg_quality', None, help="")
tf.flags.DEFINE_integer('seed', None, help="")
tf.flags.DEFINE_string('crf', None, help="")
tf.flags.DEFINE_multi_float('sigma', 0, help="")


def check_args(opt):
    if opt.crf and len(opt.sigma) != 2:
        raise ValueError("one must pass two --sigma values")


def process(image, crf=None, max_sigma=(0, 0)):
    if crf:
        i = np.random.randint(0, crf['crf'].shape[0])
        forward_crf = crf['icrf'][i]
        backward_crf = crf['crf'][i]
        irr = Noise.camera_response_function(image, forward_crf, 255)
        noise = Noise.gaussian_poisson_noise(irr,
                                             max_s=max_sigma[0],
                                             max_c=max_sigma[1])
        noisy = Noise.camera_response_function(irr + noise, backward_crf)
        noisy = np.clip(np.round(noisy * 255), 0, 255).astype('uint8')
        return noisy
    return image


def main(*args, **kwargs):
    flags = tf.flags.FLAGS
    check_args(flags)
    opt = Config()
    for key in flags:
        opt.setdefault(key, flags.get_flag_value(key, None))
    opt.steps_per_epoch = opt.num
    # set random seed at first
    np.random.seed(opt.seed)
    # check output dir
    output_dir = Path(flags.save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    writer = tf.io.TFRecordWriter(
        str(output_dir / "{}.tfrecords".format(opt.dataset)))
    data_config_file = Path(opt.data_config)
    if not data_config_file.exists():
        raise RuntimeError("dataset config file doesn't exist!")
    crf_matrix = np.load(opt.crf) if opt.crf else None
    # init loader config
    train_data, _, _ = Run.fetch_datasets(data_config_file, opt)
    train_config, _, _ = Run.init_loader_config(opt)
    loader = QuickLoader(train_data, opt.method, train_config,
                         n_threads=opt.threads, augmentation=opt.augment)
    it = loader.make_one_shot_iterator(opt.memory_limit, shuffle=True)
    with tqdm.tqdm(it, unit='batch', ascii=True) as r:
        for items in r:
            label, feature, names = items[:3]
            # label is usually HR image, feature is usually LR image
            batch_label = np.split(label, label.shape[0])
            batch_feature = np.split(feature, feature.shape[0])
            batch_name = np.split(names, names.shape[0])
            for hr, lr, name in zip(batch_label, batch_feature, batch_name):
                hr = np.squeeze(hr)
                lr = np.squeeze(lr)
                name = np.squeeze(name)
                with io.BytesIO() as fp:
                    Image.fromarray(hr, 'RGB').save(fp, format='png')
                    fp.seek(0)
                    hr_png = fp.read()
                with io.BytesIO() as fp:
                    Image.fromarray(lr, 'RGB').save(fp, format='png')
                    fp.seek(0)
                    lr_png = fp.read()
                lr_post = process(lr, crf_matrix, (opt.sigma[0], opt.sigma[1]))
                with io.BytesIO() as fp:
                    if opt.jpeg_quality:
                        Image.fromarray(lr_post, 'RGB').save(
                            fp, format='jpeg', quality=opt.jpeg_quality)
                    else:
                        Image.fromarray(lr_post, 'RGB').save(fp, format='png')
                    fp.seek(0)
                    post_png = fp.read()
                label = "{}_{}_{}".format(*name).encode()
                make_tensor_label_records(
                    [hr_png, lr_png, label, post_png],
                    ["image/hr", "image/lr", "name", "image/post"],
                    writer)


if __name__ == '__main__':
    tf.app.run(main)
