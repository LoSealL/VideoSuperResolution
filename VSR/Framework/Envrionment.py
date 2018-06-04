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
        self.li = label_index if feature_index is not None else 0

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(str(self.logdir), graph=tf.get_default_graph())

    def fit(self,
            batch,
            epochs,
            dataset,
            learning_rate=1e-4,
            learning_rate_schedule=None,
            restart=False,
            **kwargs):
        """Train the model.

        Args:
            batch: the size of mini-batch during training
            epochs: the total training epochs
            dataset: the Dataset object, used to get training and validation files
            learning_rate: the initial learning rate
            learning_rate_schedule: a callable to adjust learning rate. The signature is
                                    `fn(learning_rate, epochs, steps, loss)`
            restart: if True, start training from scratch, regardless of saved checkpoints
        """

        ckpt_last = self._find_last_ckpt() if not restart else None
        init_epoch = self._parse_ckpt_name(ckpt_last) + 1

        if ckpt_last:
            print(f'Restoring from last epoch {ckpt_last}')
            self.saver.restore(self.model.sess, str(self.savedir / ckpt_last))
        self.model.summary()
        lr = learning_rate
        max_patches = dataset.max_patches
        step_in_epoch = 0
        global_step = self.model.global_steps.eval(session=self.model.sess)
        for epoch in range(init_epoch, epochs + 1):
            dataset.setattr(max_patches=max_patches)
            loader = BatchLoader(batch, dataset, 'train', scale=self.model.scale, **kwargs)
            equal_length_mod = step_in_epoch // 20 or 10
            step_in_epoch -= step_in_epoch
            start_time = time.time()
            print(f'| Epoch: {epoch}/{epochs} |', end='')
            for img in loader:
                feature, label = img[self.fi], img[self.li]
                for fn in self.feature_callbacks:
                    feature = fn(feature)
                for fn in self.label_callbacks:
                    label = fn(label)
                self.model.train_batch(feature=feature, label=label, learning_rate=lr)
                step_in_epoch += 1
                global_step = self.model.global_steps.eval(session=self.model.sess)
                if learning_rate_schedule and callable(learning_rate_schedule):
                    lr = learning_rate_schedule(lr, epochs=epoch, steps=global_step)
                if step_in_epoch % equal_length_mod == 0:
                    print(f'=', end='', flush=True)
            consumed_time = time.time() - start_time
            print(f'| Time: {consumed_time:.4f}s, time per batch: {consumed_time * 1000 / step_in_epoch:.4f}ms/b |',
                  flush=True)
            dataset.setattr(max_patches=batch * 10)
            loader = BatchLoader(batch, dataset, 'val', scale=self.model.scale, **kwargs)
            val_metrics = {}
            for img in loader:
                feature, label = img[self.fi], img[self.li]
                for fn in self.feature_callbacks:
                    feature = fn(feature)
                for fn in self.label_callbacks:
                    label = fn(label)
                metrics, summary_op = self.model.validate_batch(feature=feature, label=label)
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k] += [v]
                self.summary_writer.add_summary(summary_op, global_step)

            for k, v in val_metrics.items():
                print(f'{k}: {np.asarray(v).mean():.4f}', end=', ')
            print('')
            ckpt_last = self._make_ckpt_name(epoch)
            self.saver.save(self.model.sess, str(self.savedir / ckpt_last))
            if self._early_exit():
                break
        # flush all pending summaries to disk
        self.summary_writer.close()
        dataset.setattr(max_patches=max_patches)

    def test(self, dataset, **kwargs):
        ckpt_last = self._find_last_ckpt()
        self.saver.restore(self.model.sess, str(self.savedir / ckpt_last))
        loader = BatchLoader(1, dataset, 'test', scale=self.model.scale, crop=False, **kwargs)
        step = 0
        for img in loader:
            feature, label = img[self.fi], img[self.li]
            for fn in self.feature_callbacks:
                feature = fn(feature)
            for fn in self.label_callbacks:
                label = fn(label)
            outputs = self.model.test_batch(feature, label)
            for fn in self.output_callbacks:
                outputs = fn(output=outputs, input=img[self.fi], label=img[self.li], step=step)
            step += 1

    def predict(self, files, mode='pil-image', depth=1, **kwargs):
        ckpt_last = self._find_last_ckpt()
        self.saver.restore(self.model.sess, str(self.savedir / ckpt_last))
        files = [Path(file) for file in to_list(files)]
        data = Dataset(test=files, mode=mode, depth=depth, **kwargs)
        loader = BatchLoader(1, data, 'test', scale=self.model.scale, crop=False, **kwargs)
        step = 0
        for img in loader:
            feature, label = img[self.fi], img[self.li]
            for fn in self.feature_callbacks:
                feature = fn(feature)
            for fn in self.label_callbacks:
                label = fn(label)
            outputs = self.model.test_batch(feature, label)
            for fn in self.output_callbacks:
                outputs = fn(output=outputs, input=feature, label=label, step=step)
            step += 1

    def export(self, export_dir='.'):
        ckpt_last = self._find_last_ckpt()
        self.saver.restore(self.model.sess, str(self.savedir / ckpt_last))
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
        ckpt = to_list(self.savedir.glob('*.ckpt.index'))
        # sort as modification time
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
        return ckpt[-1].stem if ckpt else None

    def _early_exit(self):
        return False
