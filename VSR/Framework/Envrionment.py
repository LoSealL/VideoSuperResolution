"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 11th 2018
Updated Date: May 11th 2018

SR model running environment
"""
import tensorflow as tf
import time
from pathlib import Path

from .SuperResolution import SuperResolution
from ..DataLoader.Loader import BatchLoader
from ..Util.Utility import to_list


class Environment:

    def __init__(self, model, save_dir, log_dir, **kwargs):

        assert isinstance(model, SuperResolution)

        self.model = model
        self.savedir = Path(save_dir)
        self.logdir = Path(log_dir)
        self.savedir.mkdir(parents=True, exist_ok=True)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(str(self.logdir))

    def fit(self,
            batch,
            epochs,
            dataset,
            patch_size=None,
            strides=None,
            depth=None,
            shuffle=False):

        # restore the latest checkpoint in savedir
        ckpt = to_list(self.savedir.glob('*.ckpt.index'))
        # sort as modification time
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
        ckpt_last = ckpt[-1].stem if ckpt else None
        init_epoch = self._parse_ckpt_name(ckpt_last) + 1

        if ckpt_last:
            print(f'Restoring from last epoch {init_epoch}')
            self.saver.restore(self.model.sess, str(self.savedir / ckpt_last))
        self.model.summary()

        for epoch in range(init_epoch, epochs):
            loader = BatchLoader(batch, dataset, 'train', scale=self.model.scale,
                                 patch_size=patch_size, strides=strides, depth=depth, shuffle=shuffle)
            step_in_epoch = 0
            start_time = time.time()
            for img_hr, img_lr in loader:
                self.model.train_batch(img_lr, img_hr)
                step_in_epoch += 1
                if step_in_epoch % 10 == 0:
                    print(f'=', end='', flush=True)
            consumed_time = time.time() - start_time
            print(f'| Time: {consumed_time:.4f}s, time per batch: {consumed_time * 1000 / step_in_epoch:.4f}ms/b |',
                  flush=True)
            if self._early_exit():
                break
            loader = BatchLoader(batch, dataset, 'val', scale=self.model.scale,
                                 patch_size=patch_size, strides=strides, depth=depth, shuffle=shuffle)
            val_metrics = {}
            for val_hr, val_lr in loader:
                metrics, summary_op = self.model.validate_batch(val_lr, val_hr)
                for k, v in metrics.items():
                    if not k in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k] += [v]
                self.summary_writer.add_summary(summary_op)
            for k, v in val_metrics.items():
                print(f'{k}: {sum(v) / len(v):.4f}')
            self.saver.save(self.model.sess, str(self.savedir / ckpt_last))
            ckpt_last = self._make_ckpt_name(epoch + 1)
        # flush all pending summaries to disk
        self.summary_writer.close()

    def export(self):
        pass

    def _make_ckpt_name(self, epoch):
        return f'{self.model.name}-sc{self.model.scale[0]}-ep{epoch:04d}.ckpt'

    def _parse_ckpt_name(self, name):

        # sample name: {model}-sc{scale}-ep{epoch}.ckpt(.index)
        if not name:
            return -1
        model_name, scale, epochs = Path(name).stem.split('.')[0].split('-')
        return int(epochs[2:])

    def _early_exit(self):
        return False
