"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Nov 26th 2018

Specifies a model and evaluate its corresponded checkpoint.
"""

import tensorflow as tf
import numpy as np
from . import ImagePerceptual, ImageSimilarity
from ...Util.Utility import to_list


def maybe_stack_over(data):
    data = to_list(data)
    if not data:
        return []
    try:
        if np.ndim(data[0]) <= 3:
            data = np.stack(data)
        else:
            data = np.concatenate(data)
    except ValueError:
        return data
    return [data]


def evaluate(real_images, gen_images, opt=tf.flags.FLAGS):
    if not opt.v:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    real_images = maybe_stack_over(real_images)
    gen_images = maybe_stack_over(gen_images)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)):
        tasks_to_run = []
        results = {}
        if opt.enable_psnr:
            tasks_to_run.append(ImageSimilarity.PsnrTask('PSNR'))
        if opt.enable_ssim:
            tasks_to_run.append(ImageSimilarity.SsimTask('SSIM'))
        if opt.enable_fid:
            tasks_to_run.append(ImagePerceptual.FidTask('FID'))
        if opt.enable_inception_score:
            tasks_to_run.append(ImagePerceptual.InceptionTask('InceptionScore'))
        for task in tasks_to_run:
            tf.logging.info(f"Evaluating {task.name}...")
            results[task.name] = task(real_images, gen_images)
            tf.logging.info(f"Evaluating {task.name} done\n")

    tf.logging.info("\n" + str(results))
