"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Nov 26th 2018

Specifies a model and evaluate its corresponded checkpoint.
"""

import tensorflow as tf
import numpy as np
import time
import csv
from pathlib import Path
from . import ImagePerceptual, ImageSimilarity
from ...Util.Utility import to_list


_DATE = time.strftime('%Y-%m-%d', time.localtime())
LOG_FILE = f'/tmp/vsr/{_DATE}/eval_results.csv'


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


def log_results(results, into_file=True):
    tf.logging.info("\n" + str(results))
    if not into_file:
        return
    fd = Path(LOG_FILE)
    if not fd.exists():
        fd.parent.mkdir(parents=True, exist_ok=True)
        fd = fd.open('w')
        writer = csv.DictWriter(fd, results.keys())
        writer.writeheader()
        writer.writerow(results)
    else:
        # check headers
        with fd.open('r') as f:
            checker = csv.DictReader(f)
            if checker.fieldnames != list(results.keys()):
                tf.logging.warning('header modification detected, '
                                   'write new header inline.')
                new_header = True
            else:
                new_header = False
        with fd.open('a') as f:
            writer = csv.DictWriter(f, results.keys())
            if new_header:
                writer.writeheader()
            writer.writerow(results)


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

    results.update(model=opt.model)
    log_results(results, True)
