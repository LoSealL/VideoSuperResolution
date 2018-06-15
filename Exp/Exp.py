from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import bicubic_rescale
import tensorflow as tf


class SRCNN(SuperResolution):

    def __init__(self, scale, layers=3, filters=64, kernel=(9, 5, 5), name='srcnn', **kwargs):
        self.name = name
        self.layers = layers
        self.filters = filters
        self.kernel_size = kernel
        super(SRCNN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input/lr/gray'))
            self.inputs_preproc = self.inputs
            x = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            f = self.filters
            ks = self.kernel_size
            x = self.conv2d(x, f, ks[0], activation='relu', kernel_regularizer='l2', kernel_initializer='he_normal')
            for i in range(1, self.layers - 1):
                x = self.conv2d(x, f, ks[i], activation='relu', kernel_regularizer='l2',
                                kernel_initializer='he_normal')
            x = self.conv2d(x, 1, ks[-1], kernel_regularizer='l2', kernel_initializer='he_normal')
            self.outputs.append(x)

    def build_loss(self):
        with tf.variable_scope('loss'):
            self.label.append(tf.placeholder(tf.float32, shape=[None, None, None, 1]))
            y_true = self.label[-1]
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            tv_decay = 1e-4
            tv_loss = tv_decay * tf.reduce_mean(tf.image.total_variation(y_pred))
            regular_loss = tf.losses.get_regularization_losses()
            if regular_loss != []:
                regular_loss = tf.add_n(tf.losses.get_regularization_losses())
                loss = mse + regular_loss
            else:
                loss = mse
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # self.grad = optimizer.compute_gradients(loss)
            self.loss.append(optimizer.minimize(loss, self.global_steps))
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            if regular_loss is not []:
                self.metrics['regularization'] = regular_loss
            self.metrics['tv'] = tv_loss
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        if self.metrics.get('regularization') is not None:
            tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
