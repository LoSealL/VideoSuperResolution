"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Extend the pre-Environment module, provide different and extensible
training methodology for SISR, VSR or other image tasks.
"""

import tensorflow as tf
import numpy as np
import time
import tqdm
import csv
from pathlib import Path

from ..Util.Utility import to_list
from ..Util.Config import Config


def _make_ckpt_name(name, scale, step):
    return '{}-sc{}-ep{:04d}.ckpt'.format(name, scale, step)


def _parse_ckpt_name(name):
    # sample name: {model}-sc{scale}-ep{epoch}.ckpt(.index)
    if not name:
        return 0
    model_name, scale, epochs = Path(name).stem.split('.')[0].split('-')
    return int(epochs[2:])


class Trainer:
    """A pure interface trainer.

       A trainer provides following APIs:
         >>> Trainer.fit
         >>> Trainer.infer
         >>> Trainer.benchmark
         >>> Trainer.export

       Args:
           model: the SR model object. @see SuperResolution
           work_dir: the dir to save training checkpoints and logs
           verbose: tf logging level
       """

    def __init__(self, model, work_dir, verbose=tf.logging.INFO):
        self._m = model
        self._saved = Path(work_dir) / 'save'
        self._logd = Path(work_dir) / 'log'
        self._verb = verbose
        self._restored = False
        self._csv = verbose <= tf.logging.INFO

    def _startup(self):
        tf.logging.set_verbosity(self._verb)
        self._saved.mkdir(parents=True, exist_ok=True)
        self._logd.mkdir(parents=True, exist_ok=True)
        if self._csv:
            self._csv_file = open(Path(self._logd / 'train_metrics.csv'), 'a')
            self._csv_writer = csv.writer(self._csv_file)
        if self._m.compiled:
            self.graph = tf.get_default_graph()
        else:
            with tf.Graph().as_default() as g:
                self._m.compile()
                self.graph = g

    def __enter__(self):
        """Create session of tensorflow and build model graph"""

        self._startup()
        conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(graph=self.graph, config=conf)
        sess.__enter__()
        self.savers = self._m.savers
        sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session"""

        sess = tf.get_default_session()
        sess.__exit__(exc_type, exc_val, exc_tb)

    def _find_last_ckpt(self):
        # restore the latest checkpoint in save dir
        ckpt = tf.train.get_checkpoint_state(self._saved)
        if ckpt and ckpt.model_checkpoint_path:
            return tf.train.latest_checkpoint(self._saved)
        # try another way
        ckpt = to_list(self._saved.glob('*.ckpt.index'))
        # sort as modification time
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
        return self._saved / ckpt[-1].stem if ckpt else None

    def _restore_model(self, sess):
        last_checkpoint_step = 0
        for name in self.savers:
            saver = self.savers.get(name)
            ckpt = to_list(self._saved.glob('{}*.index'.format(name)))
            if ckpt:
                ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
                ckpt = self._saved / ckpt[-1].stem
                try:
                    saver.restore(sess, str(ckpt))
                except tf.errors.NotFoundError:
                    tf.logging.warning(
                        '{} of model {} could not be restored'.format(
                            name, self._m.name))
                last_checkpoint_step = _parse_ckpt_name(ckpt)
        return last_checkpoint_step

    def _save_model(self, sess, step):
        for name in self.savers:
            saver = self.savers.get(name)
            file = self._saved / _make_ckpt_name(name, self._m.scale[0], step)
            saver.save(sess, str(file))

    def _restore(self):
        # restore graph
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('No session initialized')
        if self._restored:
            return sess
        self.last_epoch = self._restore_model(sess)
        self._restored = True
        return sess

    def export(self, export_dir='.', freeze_model=False):
        """Export model as protobuf

        Args:
            export_dir: directory to save the exported model
            freeze_model: freeze all trainable variables
        """

        self._restore()
        if freeze_model:
            self._m.export_freeze_model(export_dir)
        else:
            self._m.export_saved_model(export_dir)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def benchmark(self, *args, **kwargs):
        raise NotImplementedError


class VSR(Trainer):
    """Default trainer for task SISR or VSR"""
    v = Config()  # local variables
    """=======================================
        components, sub-functions, helpers
       =======================================
    """

    def query_config(self, config, **kwargs) -> Config:
        assert isinstance(config, Config)
        config.update(kwargs)  # override parameters
        self.v.epoch = config.epoch  # current epoch
        self.v.epochs = config.epochs  # total epochs
        self.v.lr = config.lr  # learning rate
        self.v.lr_schedule = config.lr_schedule
        self.v.memory_limit = config.memory_limit
        self.v.feature_callbacks = config.feature_callbacks or []
        self.v.label_callbacks = config.label_callbacks or []
        self.v.output_callbacks = config.output_callbacks or []
        self.v.validate_every_n_epoch = config.validate_every_n_epoch or 1
        self.v.subdir = config.subdir
        return self.v

    def fit_init(self) -> bool:
        v = self.v
        v.sess = self._restore()
        if self.last_epoch >= v.epochs:
            return False
        tf.logging.info('Fitting: {}'.format(self._m.name.upper()))
        self._m.display()
        v.summary_writer = tf.summary.FileWriter(
            str(self._logd), graph=tf.get_default_graph())
        v.global_step = self._m.global_steps.eval()
        return True

    def fit_close(self):
        # flush all pending summaries to disk
        if self.v.summary_writer:
            self.v.summary_writer.close()
        if self._csv:
            self._csv_file.close()

    def fn_train_each_epoch(self):
        v = self.v
        train_iter = v.train_loader.make_one_shot_iterator(
            v.memory_limit, shuffle=True)
        if hasattr(v.train_loader, 'prefetch'):
            v.train_loader.prefetch(v.memory_limit)
        date = time.strftime('%Y-%m-%d %T', time.localtime())
        v.avg_meas = {}
        if v.lr_schedule and callable(v.lr_schedule):
            v.lr = v.lr_schedule(steps=v.global_step)
        print('| {} | Epoch: {}/{} | LR: {:.2g} |'.format(
            date, v.epoch, v.epochs, v.lr))
        with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
            for label, feature, name in r:
                self.fn_train_each_step(label, feature, name)
                r.set_postfix(v.loss)
        for _k, _v in v.avg_meas.items():
            print('| Epoch average {} = {:.6f} |'.format(_k, np.mean(_v)))
        if self._csv:
            if self._csv_file.tell() == 0:
                self._csv_writer.writerow(v.avg_meas.keys())
            self._csv_writer.writerow([np.mean(s) for s in v.avg_meas.values()])
            self._csv_file.flush()
        if v.epoch % v.validate_every_n_epoch == 0:
            self.benchmark(v.val_loader, v, epoch=v.epoch)
            v.summary_writer.add_summary(self._m.summary(), v.global_step)
            self._save_model(v.sess, v.epoch)

    def fn_train_each_step(self, label=None, feature=None, name=None):
        v = self.v
        for fn in v.feature_callbacks:
            feature = fn(feature, name=name)
        for fn in v.label_callbacks:
            label = fn(label, name=name)
        loss = self._m.train_batch(feature, label, learning_rate=v.lr,
                                   epochs=v.epoch)
        v.global_step = self._m.global_steps.eval()
        for _k, _v in loss.items():
            v.avg_meas[_k] = \
                v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
            loss[_k] = '{:08.5f}'.format(_v)
        v.loss = loss

    def fn_infer_each_step(self, label=None, feature=None, name=None):
        v = self.v
        origin_feat = feature
        for fn in v.feature_callbacks:
            feature = fn(feature, name=name)
        outputs, _ = self._m.test_batch(feature, None)
        for fn in v.output_callbacks:
            outputs = fn(outputs, input=origin_feat, name=name,
                         subdir=v.subdir, mode=v.color_format)

    def fn_benchmark_each_step(self, label=None, feature=None, name=None):
        v = self.v
        origin_feat = feature
        for fn in v.feature_callbacks:
            feature = fn(feature, name=name)
        for fn in v.label_callbacks:
            label = fn(label, name=name)
        outputs, metrics = self._m.test_batch(feature, label, epochs=v.epoch)
        for _k, _v in metrics.items():
            if _k not in v.mean_metrics:
                v.mean_metrics[_k] = []
            v.mean_metrics[_k] += [_v]
        for fn in v.output_callbacks:
            outputs = fn(outputs, input=origin_feat, label=label, name=name,
                         mode=v.color_format, subdir=v.subdir)

    def fn_benchmark_body(self):
        v = self.v
        it = v.loader.make_one_shot_iterator(v.memory_limit, shuffle=False)
        for label, feature, name in tqdm.tqdm(it, 'Test', ascii=True):
            self.fn_benchmark_each_step(label, feature, name)

    """=======================================
        Interface: fit, benchmark, infer
       =======================================
    """

    def fit(self, loaders, config, **kwargs):
        """Fit the model.

        Args:
            loaders: a tuple of 2 loaders, the 1st one is used for training,
              and the 2nd one is used for validating.
            config: fitting configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        v = self.query_config(config, **kwargs)
        v.train_loader, v.val_loader = loaders
        if not self.fit_init():
            return
        for epoch in range(self.last_epoch + 1, v.epochs + 1):
            v.epoch = epoch
            self.fn_train_each_epoch()
        self.fit_close()

    def infer(self, loader, config, **kwargs):
        """Infer SR images.

        Args:
            loader: a loader for enumerating LR images
            config: inferring configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        v = self.query_config(config, **kwargs)
        v.color_format = loader.color_format

        self._restore()
        it = loader.make_one_shot_iterator()
        if len(it):
            tf.logging.info('Inferring {} at epoch {}'.format(
                self._m.name, self.last_epoch))
        else:
            return
        # use original images in inferring
        for feature, _, name in tqdm.tqdm(it, 'Infer', ascii=True):
            self.fn_infer_each_step(None, feature, name)

    def benchmark(self, loader, config, **kwargs):
        """Benchmark/validate the model.

        Args:
            loader: a loader for enumerating LR images
            config: benchmark configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        v = self.query_config(config, **kwargs)
        v.color_format = loader.color_format

        self._restore()
        v.mean_metrics = {}
        v.loader = loader
        self.fn_benchmark_body()
        for _k, _v in v.mean_metrics.items():
            print('{}: {:.6f}'.format(_k, np.mean(_v)), end=', ')
        print('')
