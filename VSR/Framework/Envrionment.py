"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 11th 2018
Updated Date: May 17th 2018

SR model running environment
"""
import tensorflow as tf
import numpy as np
import time
from pathlib import Path

from .SuperResolution import SuperResolution
from ..DataLoader.Loader import BatchLoader
from ..DataLoader.Dataset import Dataset
from ..Util.Utility import to_list


class Environment:
    """A model wrapper like tf.Estimator"""

    def __init__(self,
                 model,
                 save_dir,
                 log_dir,
                 feature_callbacks=None,
                 label_callbacks=None,
                 output_callbacks=None,
                 feature_index=None,
                 label_index=None,
                 **kwargs):
        """Initiate the object

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
        """
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

    def __enter__(self):
        """Create session of tensorflow and build model graph"""

        sess = tf.Session()
        sess.__enter__()
        if not self.model.compiled:
            self.model.compile()
        self.saver = tf.train.Saver(max_to_keep=10, allow_empty=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session and
        ? clear graph ?
        """

        sess = tf.get_default_session()
        sess.__exit__(exc_type, exc_val, exc_tb)
        # tf.reset_default_graph() ?

    def fit(self,
            batch=32,
            epochs=1,
            dataset=None,
            learning_rate=1e-4,
            learning_rate_schedule=None,
            restart=False,
            validate_numbers=1,
            validate_every_n_epoch=1,
            **kwargs):
        """Train the model.

        Args:
            batch: the size of mini-batch during training
            epochs: the total training epochs
            dataset: the Dataset object, used to get training and validation frames
            learning_rate: the initial learning rate
            learning_rate_schedule: a callable to adjust learning rate. The signature is
                                    `fn(learning_rate, epochs, steps, loss)`
            restart: if True, start training from scratch, regardless of saved checkpoints
            validate_every_n_epoch: run validation every n epochs
        """

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('No session initialized')
        if not self.model.compiled:
            print('[Warning] model not compiled, compiling now...')
            self.model.compile()
        sess.run(tf.global_variables_initializer())
        ckpt_last = self._find_last_ckpt() if not restart else None
        init_epoch = self._parse_ckpt_name(ckpt_last) + 1
        if init_epoch > epochs:
            return
        # saver = tf.train.Saver(var_list=tf.trainable_variables('psrn/ResGen'))
        if ckpt_last:
            print(f'Restoring from last epoch {ckpt_last}')
            self.saver.restore(sess, str(ckpt_last))
        self.model.summary()
        summary_writer = tf.summary.FileWriter(str(self.logdir), graph=tf.get_default_graph())
        lr = learning_rate
        global_step = self.model.global_steps.eval()
        if learning_rate_schedule and callable(learning_rate_schedule):
            lr = learning_rate_schedule(lr, epochs=init_epoch, steps=global_step)
        train_loader = BatchLoader(batch, dataset, 'train', scale=self.model.scale, **kwargs)
        dataset.setattr(random=True, max_patches=batch * validate_numbers)
        val_loader = BatchLoader(batch, dataset, 'val', scale=self.model.scale, crop=True, **kwargs)
        for epoch in range(init_epoch, epochs + 1):
            train_loader.reset()
            total_steps = len(train_loader)
            equal_length_mod = max(total_steps // 20, 1)
            step_in_epoch = 0
            start_time = time.time()
            date = time.strftime('%Y-%m-%D %T', time.localtime())
            print(f'| {date} | Epoch: {epoch}/{epochs} | LR: {lr} |')
            avg_meas = {}
            for img in train_loader:
                feature, label, name = img[self.fi], img[self.li], str(img[-1])
                for fn in self.feature_callbacks:
                    feature = fn(feature, name=name)
                for fn in self.label_callbacks:
                    label = fn(label, name=name)
                loss = self.model.train_batch(feature=feature, label=label, learning_rate=lr, epochs=epoch)
                step_in_epoch += 1
                global_step = self.model.global_steps.eval()
                if learning_rate_schedule and callable(learning_rate_schedule):
                    lr = learning_rate_schedule(lr, epochs=epoch, steps=global_step)
                n_equals = step_in_epoch // equal_length_mod
                n_dots = total_steps // equal_length_mod - n_equals
                bar = f'{step_in_epoch}/{total_steps} [' + '=' * n_equals + '.' * n_dots + ']'
                for k, v in loss.items():
                    avg_meas[k] = avg_meas[k] + v if avg_meas.get(k) else v
                    bar += f' {k}={v:.4f}'
                print(bar, flush=True, end='\r')
            consumed_time = time.time() - start_time
            print()
            for k, v in avg_meas.items():
                print(f'| Epoch average {k} = {v / step_in_epoch:.6f} |')
            print(f'| Time: {consumed_time:.4f}s, time per batch: {consumed_time * 1e3 / step_in_epoch:.4f}ms/b |', flush=True)

            if epoch % validate_every_n_epoch:
                continue
            val_metrics = {}
            val_loader.reset()
            for img in val_loader:
                feature, label, name = img[self.fi], img[self.li], str(img[-1])
                for fn in self.feature_callbacks:
                    feature = fn(feature, name=name)
                for fn in self.label_callbacks:
                    label = fn(label, name=name)
                metrics, val_summary_op = self.model.validate_batch(feature=feature, label=label, epochs=epoch)
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k] += [v]
                summary_writer.add_summary(val_summary_op, global_step)
            for k, v in val_metrics.items():
                print(f'{k}: {np.asarray(v).mean():.6f}', end=', ')
            print('')
            ckpt_last = self._make_ckpt_name(epoch)
            self.saver.save(sess, str(self.savedir / ckpt_last))
            if self._early_exit():
                break
        # flush all pending summaries to disk
        summary_writer.close()

    def test(self, dataset, **kwargs):
        r"""Test model with test sets in dataset
        
        Args:
            dataset: instance of dataset, with dataset.test valid
        """

        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        ckpt_last = self._find_last_ckpt()
        print('===================================')
        print(f'Testing model: {self.model.name} by {ckpt_last}')
        print('===================================')
        self.saver.restore(sess, str(ckpt_last))
        loader = BatchLoader(1, dataset, 'test', scale=self.model.scale, crop=False, **kwargs)
        for img in loader:
            feature, label, name = img[self.fi], img[self.li], str(img[-1])
            for fn in self.feature_callbacks:
                feature = fn(feature, name=name)
            for fn in self.label_callbacks:
                label = fn(label, name=name)
            outputs = self.model.test_batch(feature, None)
            for fn in self.output_callbacks:
                outputs = fn(outputs, input=img[self.fi], label=img[self.li], name=name, mode=loader.color_format)

    def predict(self, files, mode='pil-image1', depth=1, **kwargs):
        r"""Predict output for frames

        Args:
            files: a list of frames as inputs
            mode: specify file format. `pil-image1` for PIL supported images, or `NV12/YV12/RGB` for raw data
            depth: specify length of sequence of images. 1 for images, >1 for videos
        """

        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        ckpt_last = self._find_last_ckpt()
        self.saver.restore(sess, str(ckpt_last))
        files = [Path(file) for file in to_list(files)]
        data = Dataset(test=files, mode=mode, depth=depth, modcrop=False, **kwargs)
        loader = BatchLoader(1, data, 'test', scale=self.model.scale, crop=False, **kwargs)
        for img in loader:
            feature, label, name = img[self.fi], img[self.li], str(img[-1])
            for fn in self.feature_callbacks:
                feature = fn(feature, name=name)
            outputs = self.model.test_batch(feature, None)
            for fn in self.output_callbacks:
                outputs = fn(outputs, input=img[self.fi], label=img[self.li], name=name, mode=loader.color_format)

    def export(self, export_dir='.'):
        """Export model as protobuf
        
        Args:
            export_dir: directory to save the exported model
        """

        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        ckpt_last = self._find_last_ckpt()
        self.saver.restore(sess, str(ckpt_last))
        self.model.export_model_pb(export_dir)

    def _make_ckpt_name(self, epoch):
        return f'{self.model.name}-sc{self.model.scale[0]}-ep{epoch:04d}.ckpt'

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

    def _early_exit(self):
        # Todo not implemented
        return False
