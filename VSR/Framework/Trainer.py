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

    def _startup(self):
        tf.logging.set_verbosity(self._verb)
        self._saved.mkdir(parents=True, exist_ok=True)
        self._logd.mkdir(parents=True, exist_ok=True)
        if self._m.compiled:
            self.graph = tf.get_default_graph()
        else:
            with tf.Graph().as_default() as g:
                self._m.compile()
                self.graph = g

    def __enter__(self):
        """Create session of tensorflow and build model graph"""

        self._startup()
        sess = tf.Session(graph=self.graph)
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
                    tf.logging.warning('{} of model {} could not be restored'.format(name, self._m.name))
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

    def export(self, export_dir='.'):
        """Export model as protobuf

        Args:
            export_dir: directory to save the exported model
        """

        self._restore()
        self._m.export_saved_model(export_dir)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def benchmark(self, *args, **kwargs):
        raise NotImplementedError


class VSR(Trainer):
    """Default trainer for task SISR or VSR"""

    def fit(self, loaders, config, **kwargs):
        """Fit the model.

        Args:
            loaders: a tuple of 2 loaders, the 1st one is used for training,
              and the 2nd one is used for validating.
            config: fitting configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        assert isinstance(config, Config)
        config.update(kwargs)
        epochs = config.epochs
        lr = config.lr
        lr_schedule = config.lr_schedule
        memory_usage = config.memory_limit
        feature_callbacks = config.feature_callbacks or []
        label_callbacks = config.label_callbacks or []
        validate_every_n_epoch = config.validate_every_n_epoch or 1

        sess = self._restore()
        if self.last_epoch >= epochs:
            return
        tf.logging.info('Fitting: {}'.format(self._m.name.upper()))
        self._m.display()
        summary_writer = tf.summary.FileWriter(str(self._logd), graph=tf.get_default_graph())
        train_loader, val_loader = loaders
        global_step = self._m.global_steps.eval()

        for epoch in range(self.last_epoch + 1, epochs + 1):
            train_iter = train_loader.make_one_shot_iterator(memory_usage, shuffle=True)
            if hasattr(train_loader, 'prefetch'):
                train_loader.prefetch(memory_usage)
            date = time.strftime('%Y-%m-%d %T', time.localtime())
            print('| {} | Epoch: {}/{} | LR: {:.2g} |'.format(date, epoch, epochs, lr))
            avg_meas = {}
            if lr_schedule and callable(lr_schedule):
                lr = lr_schedule(lr, epochs=epoch, steps=global_step)
            with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
                for label, feature, name in r:
                    for fn in feature_callbacks:
                        feature = fn(feature, name=name)
                    for fn in label_callbacks:
                        label = fn(label, name=name)
                    loss = self._m.train_batch(feature, label, learning_rate=lr, epochs=epoch)
                    global_step = self._m.global_steps.eval()
                    for k, v in loss.items():
                        avg_meas[k] = avg_meas[k] + [v] if avg_meas.get(k) else [v]
                        loss[k] = '{:08.5f}'.format(v)
                    r.set_postfix(loss)
            for k, v in avg_meas.items():
                print('| Epoch average {} = {:.6f} |'.format(k, np.mean(v)))

            if epoch % validate_every_n_epoch:
                continue
            self.benchmark(val_loader, config, epoch=epoch)
            summary_writer.add_summary(self._m.summary(), global_step)
            self._save_model(sess, epoch)
        # flush all pending summaries to disk
        summary_writer.close()

    def infer(self, loader, config, **kwargs):
        """Infer SR images.

        Args:
            loader: a loader for enumerating LR images
            config: inferring configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        assert isinstance(config, Config)
        config.update(kwargs)
        feature_callbacks = config.feature_callbacks or []
        output_callbacks = config.output_callbacks or []

        self._restore()
        it = loader.make_one_shot_iterator()
        if len(it):
            tf.logging.info('Inferring {} at epoch {}'.format(self._m.name, self.last_epoch))
        else:
            return
        # use original images in inferring
        for feature, _, name in tqdm.tqdm(it, 'Infer', ascii=True):
            for fn in feature_callbacks:
                feature = fn(feature, name=name)
            outputs, _ = self._m.test_batch(feature, None)
            for fn in output_callbacks:
                outputs = fn(outputs, input=feature, mode=loader.color_format, name=name)

    def benchmark(self, loader, config, **kwargs):
        """Benchmark/validate the model.

        Args:
            loader: a loader for enumerating LR images
            config: benchmark configuration, an instance of `Util.Config.Config`
            kwargs: additional arguments to override the same ones in config.
        """
        assert isinstance(config, Config)
        config.update(kwargs)
        epoch = config.epoch
        memory_usage = config.memory_limit
        subdir = config.subdir
        feature_callbacks = config.feature_callbacks or []
        label_callbacks = config.label_callbacks or []
        output_callbacks = config.output_callbacks or []

        self._restore()
        it = loader.make_one_shot_iterator(memory_usage, shuffle=False)
        mean_metrics = {}
        for label, feature, name in tqdm.tqdm(it, 'Test', ascii=True):
            for fn in feature_callbacks:
                feature = fn(feature, name=name)
            for fn in label_callbacks:
                label = fn(label, name=name)
            outputs, metrics = self._m.test_batch(feature, label, epochs=epoch)
            for k, v in metrics.items():
                if k not in mean_metrics:
                    mean_metrics[k] = []
                mean_metrics[k] += [v]
            for fn in output_callbacks:
                outputs = fn(outputs, input=feature, label=label, mode=loader.color_format,
                             name=name, subdir=subdir)
        for k, v in mean_metrics.items():
            print('{}: {:.6f}'.format(k, np.mean(v)), end=', ')
        print('')
