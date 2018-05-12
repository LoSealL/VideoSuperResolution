import tensorflow as tf


def add_summaries(scope_name, model_name, var, save_stddev=True, save_mean=False, save_max=False, save_min=False):
    with tf.name_scope(scope_name):
        mean_var = tf.reduce_mean(var)
        if save_mean:
            tf.summary.scalar("mean/" + model_name, mean_var)

        if save_stddev:
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))
            tf.summary.scalar("stddev/" + model_name, stddev_var)

        if save_max:
            tf.summary.scalar("max/" + model_name, tf.reduce_max(var))

        if save_min:
            tf.summary.scalar("min/" + model_name, tf.reduce_min(var))
        tf.summary.histogram(model_name, var)
