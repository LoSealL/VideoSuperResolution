''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.
See --help to see further details.
'''

from pathlib import Path
import tensorflow as tf


def preprocess_for_inception(images):
    """Preprocess images for inception.

    Args:
      images: images minibatch. Shape [batch size, width, height,
        channels]. Values are in [0..255].

    Returns:
      preprocessed_images
    """

    images = tf.cast(images, tf.float32)

    # tfgan_eval.preprocess_image function takes values in [0, 255]
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.0),
                                  tf.assert_less_equal(images, 255.0)]):
        images = tf.identity(images)

    preprocessed_images = tf.map_fn(
        fn=tf.contrib.gan.eval.preprocess_image,
        elems=images,
        back_prop=False)

    return preprocessed_images


def fid_score(real_image, gen_image, inception_graph):
    if isinstance(inception_graph, (str, Path)) and Path(inception_graph).exists():
        graph_def = tf.contrib.gan.eval.get_graph_def_from_disk(inception_graph)
    else:
        graph_def = inception_graph
    feature_real = tf.contrib.gan.eval.run_inception(
        preprocess_for_inception(real_image),
        graph_def=graph_def,
        output_tensor='pool_3:0')
    feature_gen = tf.contrib.gan.eval.run_inception(
        preprocess_for_inception(gen_image),
        graph_def=graph_def,
        output_tensor='pool_3:0')
    fid = tf.contrib.gan.eval.frechet_classifier_distance(
        classifier_fn=tf.identity,
        real_images=feature_real,
        generated_images=feature_gen,
        num_batches=1)
    return fid
