"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 11th 2018
Updated Date: May 17th 2018

SR model running environment
"""
import tensorflow as tf
import numpy as np
import time
import tqdm
from pathlib import Path

from VSR.Framework.SuperResolution import SuperResolution
from VSR.DataLoader.Loader import MpLoader, QuickLoader
from VSR.DataLoader.Dataset import Dataset
from VSR.Util.Utility import to_list


class Environment:
    """A model wrapper like tf.Estimator

    Args:
        model: the compiled model object. Have to call model.compile explicitly
        save_dir: the dir to save training checkpoints
        log_dir: the dir to save tensorboard log events
        feature_callbacks: a list of callable called in turn to process features.
                           the signature of this callable is `fn(x)->x`
        label_callbacks: a list of callable called in turn to process labels
                         the signature of this callable is `fn(x)->x`
        output_callbacks: a list of callable called in turn to post-process outputs
                          the signature of this callable is `fn(input, output, save_dir, step)->output`
        feature_index: the index to access in the return list of BatchLoader, default the 2nd item in list
        label_index: the index to access in the return list of BatchLoader, default the 1st item in list
        verbose: tf logger level
    """

    def __init__(self,
                 model,
                 save_dir,
                 log_dir,
                 feature_callbacks=None,
                 label_callbacks=None,
                 output_callbacks=None,
                 feature_index=None,
                 label_index=None,
                 verbose=tf.logging.INFO,
                 **kwargs):
        assert isinstance(model, SuperResolution)

        self.model = model
        self.savedir = Path(save_dir)
        self.logdir = Path(log_dir)
        self.savedir.mkdir(parents=True, exist_ok=True)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.feature_callbacks = feature_callbacks or []
        self.label_callbacks = label_callbacks or []
        self.output_callbacks = output_callbacks or []
        self.fi = feature_index if feature_index is not None else 1
        self.li = label_index if label_index is not None else 0
        tf.logging.set_verbosity(verbose)

    def __enter__(self):
        """Create session of tensorflow and build model graph"""

        sess = tf.Session()
        sess.__enter__()
        if not self.model.compiled:
            self.model.compile()
        self.savers = self.model.savers
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session and
        #TODO ? clear graph ?
        """

        sess = tf.get_default_session()
        sess.__exit__(exc_type, exc_val, exc_tb)
        # tf.reset_default_graph() ?

    def _make_ckpt_name(self, name, step):
        return f'{name}-sc{self.model.scale[0]}-ep{step:04d}.ckpt'

    def _parse_ckpt_name(self, name):
        # sample name: {model}-sc{scale}-ep{epoch}.ckpt(.index)
        if not name:
            return 0
        model_name, scale, epochs = Path(name).stem.split('.')[0].split('-')
        return int(epochs[2:])

    def _find_last_ckpt(self):
        # restore the latest checkpoint in savedir
        ckpt = tf.train.get_checkpoint_state(self.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            return tf.train.latest_checkpoint(self.savedir)
        # try another way
        ckpt = to_list(self.savedir.glob('*.ckpt.index'))
        # sort as modification time
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
        return self.savedir / ckpt[-1].stem if ckpt else None

    def _restore_model(self, sess):
        last_checkpoint_step = 0
        for name in self.savers:
            saver = self.savers.get(name)
            ckpt = to_list(self.savedir.glob(f'{name}*.index'))
            if ckpt:
                ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
                ckpt = self.savedir / ckpt[-1].stem
                try:
                    saver.restore(sess, str(ckpt))
                except:
                    tf.logging.warning(f'{name} of model {self.model.name} counld not be restored')
                last_checkpoint_step = self._parse_ckpt_name(ckpt)
        return last_checkpoint_step

    def _save_model(self, sess, step):
        for name in self.savers:
            saver = self.savers.get(name)
            file = self.savedir / self._make_ckpt_name(name, step)
            saver.save(sess, str(file))

    def _early_exit(self):
        # Todo not implemented
        return False

    def fit(self,
            batch=32,
            epochs=1,
            steps_per_epoch=200,
            dataset=None,
            learning_rate=1e-4,
            learning_rate_schedule=None,
            restart=False,
            validate_numbers=1,
            validate_every_n_epoch=1,
            augmentation=False,
            parallel=1,
            memory_usage=None,
            **kwargs):
        """Train the model.

        Args:
            batch: the size of mini-batch during training
            epochs: the total training epochs
            steps_per_epoch: training steps of each epoch
            dataset: the Dataset object, used to get training and validation frames
            learning_rate: the initial learning rate
            learning_rate_schedule: a callable to adjust learning rate. The signature is
                                    `fn(learning_rate, epochs, steps, loss)`
            restart: if True, start training from scratch, regardless of saved checkpoints
            validate_numbers: the number of patches in validation
            validate_every_n_epoch: run validation every n epochs
            augmentation: a boolean representing conduct image augmentation (random flip and rotate)
            parallel: a scalar representing threads number of loading dataset
            memory_usage: a string or integer, limiting maximum usage of CPU memory. (i.e. 1024MB, 8GB...)
        """

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('No session initialized')
        if not self.model.compiled:
            tf.logging.error('[Warning] model not compiled, compiling now...')
            self.model.compile()
        sess.run(tf.global_variables_initializer())
        init_epoch = 1 if restart else self._restore_model(sess) + 1
        if init_epoch > epochs:
            return
        print('===================================')
        print(f'Training model: {self.model.name.upper()}')
        print('===================================')
        self.model.display()
        summary_writer = tf.summary.FileWriter(str(self.logdir), graph=tf.get_default_graph())
        lr = learning_rate
        global_step = self.model.global_steps.eval()
        if learning_rate_schedule and callable(learning_rate_schedule):
            lr = learning_rate_schedule(lr, epochs=init_epoch, steps=global_step)
        if parallel == 1:
            train_loader = QuickLoader(batch, dataset, 'train', self.model.scale, steps_per_epoch,
                                       crop='random', augmentation=augmentation, **kwargs)
        else:
            train_loader = MpLoader(batch, dataset, 'train', self.model.scale, steps_per_epoch,
                                    crop='random', augmentation=augmentation, **kwargs)
        val_loader = QuickLoader(batch, dataset, 'val', self.model.scale, validate_numbers, crop='center', **kwargs)
        for epoch in range(init_epoch, epochs + 1):
            train_iter = train_loader.make_one_shot_iterator(memory_usage, shard=parallel, shuffle=True)
            date = time.strftime('%Y-%m-%d %T', time.localtime())
            print(f'| {date} | Epoch: {epoch}/{epochs} | LR: {lr} |')
            avg_meas = {}
            with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
                for img in r:
                    feature, label, name = img[self.fi], img[self.li], img[-1]
                    for fn in self.feature_callbacks:
                        feature = fn(feature, name=name)
                    for fn in self.label_callbacks:
                        label = fn(label, name=name)
                    loss = self.model.train_batch(
                        feature=feature, label=label, learning_rate=lr, epochs=epoch)
                    global_step = self.model.global_steps.eval()
                    if learning_rate_schedule and callable(learning_rate_schedule):
                        lr = learning_rate_schedule(lr, epochs=epoch, steps=global_step)
                    for k, v in loss.items():
                        avg_meas[k] = avg_meas[k] + [v] if avg_meas.get(k) else [v]
                        loss[k] = '{:08.5f}'.format(v)
                    r.set_postfix(loss)
            for k, v in avg_meas.items():
                print(f'| Epoch average {k} = {np.mean(v):.6f} |')

            if epoch % validate_every_n_epoch: continue
            val_metrics = {}
            val_iter = val_loader.make_one_shot_iterator(memory_usage, shard=parallel, shuffle=False)
            for img in val_iter:
                feature, label, name = img[self.fi], img[self.li], img[-1]
                for fn in self.feature_callbacks:
                    feature = fn(feature, name=name)
                for fn in self.label_callbacks:
                    label = fn(label, name=name)
                metrics, val_summary_op, _ = self.model.validate_batch(
                    feature=feature, label=label, epochs=epoch)
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k] += [v]
                summary_writer.add_summary(val_summary_op, global_step)
            for k, v in val_metrics.items():
                print(f'{k}: {np.mean(v):.6f}', end=', ')
            print('')
            self._save_model(sess, epoch)
        # flush all pending summaries to disk
        summary_writer.close()

    def test(self, dataset, **kwargs):
        r"""Test model with test sets in dataset
        
        Args:
            dataset: instance of dataset, with dataset.test valid
        """

        sess = tf.get_default_session()
        ckpt_last = self._restore_model(sess)
        loader = QuickLoader(1, dataset, 'test', self.model.scale, -1, crop=None, **kwargs)
        it = loader.make_one_shot_iterator()
        if len(it):
            print('===================================')
            print(f'Testing model: {self.model.name} by {ckpt_last}')
            print('===================================')
        else:
            return
        for img in tqdm.tqdm(it, 'Test', ascii=True):
            feature, label, name = img[self.fi], img[self.li], img[-1]
            tf.logging.debug('output: ' + str(name))
            for fn in self.feature_callbacks:
                feature = fn(feature, name=name)
            for fn in self.label_callbacks:
                label = fn(label, name=name)
            outputs = self.model.test_batch(feature, None)
            for fn in self.output_callbacks:
                outputs = fn(outputs, input=img[self.fi], label=img[self.li], mode=loader.color_format,
                             name=name, subdir=dataset.name)

    def predict(self, files, mode='pil-image1', depth=1, **kwargs):
        r"""Predict output for frames

        Args:
            files: a list of frames as inputs
            mode: specify file format. `pil-image1` for PIL supported images, or `NV12/YV12/RGB` for raw data
            depth: specify length of sequence of images. 1 for images, >1 for videos
        """

        sess = tf.get_default_session()
        ckpt_last = self._restore_model(sess)
        files = [Path(file) for file in to_list(files)]
        data = Dataset(test=files, mode=mode, depth=depth, modcrop=False, **kwargs)
        loader = QuickLoader(1, data, 'test', self.model.scale, -1, crop=None, **kwargs)
        it = loader.make_one_shot_iterator()
        if len(it):
            print('===================================')
            print(f'Predicting model: {self.model.name} by {ckpt_last}')
            print('===================================')
        else:
            return
        for img in tqdm.tqdm(it, 'Infer', ascii=True):
            feature, label, name = img[self.fi], img[self.li], img[-1]
            tf.logging.debug('output: ' + str(name))
            for fn in self.feature_callbacks:
                feature = fn(feature, name=name)
            outputs = self.model.test_batch(feature, None)
            for fn in self.output_callbacks:
                outputs = fn(outputs, input=img[self.fi], label=img[self.li], mode=loader.color_format, name=name)

    def export(self, export_dir='.'):
        """Export model as protobuf
        
        Args:
            export_dir: directory to save the exported model
        """

        sess = tf.get_default_session()
        self._restore_model(sess)
        self.model.export_saved_model(export_dir)
