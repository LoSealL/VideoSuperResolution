"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 17th 2018

Training environment callbacks preset
"""

from pathlib import Path
from functools import partial
import numpy as np
from PIL.Image import Image

from ..Util.ImageProcess import array_to_img, img_to_array, imresize


def _sub_residual(**kwargs):
    img = kwargs.get('input')
    res = kwargs.get('output') or np.zeros_like(img)
    res = res[0] if isinstance(res, list) else res
    return img - res


def _save_model_predicted_images(output, index, mode='YCbCr', **kwargs):
    save_dir = kwargs.get('save_dir') or '.'
    name = kwargs.get('name')
    if output is not None:
        img = output[index] if isinstance(output, list) else output
        img = _to_normalized_image(img, mode)
        path = Path(f'{save_dir}/{name}_PR.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        rep = 1
        while path.exists():
            path = Path(f'{save_dir}/{name}_PR_{rep}.png')
            rep += 1
        img.convert('RGB').save(str(path))
    return output


def _colored_grayscale_image(outputs, input, **kwargs):
    ret = []
    for img in outputs:
        assert img.shape[-1] == 1
        scale = np.array(img.shape[1:3]) // np.array(input.shape[1:3])
        uv = array_to_img(input[0], 'YCbCr')
        uv = imresize(uv, scale)
        uv = img_to_array(uv)[..., 1:]
        img = np.concatenate([img[0], uv], axis=-1)
        img = np.clip(img, 0, 255)
        ret.append(array_to_img(img, 'YCbCr'))
    return ret


def _to_normalized_image(img, mode):
    img = np.asarray(img)
    # squeeze to [H, W, C]
    try:
        img = np.squeeze(img)
    except ValueError:
        pass
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError('Invalid img data, must be an array of 2D image1 with channel less than 3')
    if img.shape[-1] == 2:
        # treat 2 channels image as optical flow
        return _flow_to_image(img, mode)
    img = np.clip(img, 0, 255)
    return array_to_img(img, mode)


def _color_wheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    color = np.zeros([RY + YG + GC + CB + BM + MR, 3])
    for i in range(RY):
        color[i] = [255, 255 * i / RY, 0]
    for i in range(YG):
        color[i + RY] = [255 - 255 * i / YG, 255, 0]
    for i in range(GC):
        color[i + RY + YG] = [0, 255, 255 * i / GC]
    for i in range(CB):
        color[i + RY + YG + GC] = [0, 255 - 255 * i / CB, 255]
    for i in range(BM):
        color[i + RY + YG + GC + CB] = [255 * i / BM, 0, 255]
    for i in range(MR):
        color[i + RY + YG + GC + CB + BM] = [255, 0, 255 - 255 * i / MR]
    return color / 255


def _viz_flow(u, v, logscale=True, scaledown=6):
    """
    Copied from https://github.com/jswulff/pcaflow/blob/master/pcaflow/utils/viz_flow.py
    thanks to @jswulff

    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = _color_wheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u ** 2 + v ** 2)
    if logscale:
        radius = np.log(radius + 1)
    radius = radius / scaledown
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., ncols

    k1 = k0 + 1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape + (ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1 - f) * col0 + f * col1

        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        # out of range
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)


def _flow_to_image(flow, mode):
    u = flow[..., 0]
    v = flow[..., 1]
    viz = _viz_flow(u, v)
    return array_to_img(viz, 'RGB')


def _add_noise(feature, stddev, mean, clip, **kwargs):
    x = feature.astype('float') + np.random.normal(mean, stddev, feature.shape)
    return np.clip(x, 0, 255) if clip else x


def _add_random_noise(feature, low, high, step, mean, clip, **kwargs):
    n = list(range(low, high, step))
    i = np.random.randint(len(n))
    stddev = n[i]
    return _add_noise(feature, stddev, mean, clip)


def _gaussian_blur(feature, width, size, **kwargs):
    from scipy.ndimage.filters import gaussian_filter as gf

    y = []
    for img in np.split(feature, feature.shape[0]):
        c = []
        for channel in np.split(img, img.shape[-1]):
            channel = np.squeeze(channel).astype('float')
            c.append(gf(channel, width, mode='constant', truncate=(size // 2) / width))
        y.append(np.stack(c, axis=-1))
    return np.stack(y)


def _exponential_decay(lr, start_lr, epochs, steps, decay_step, decay_rate):
    return start_lr * decay_rate ** (steps / decay_step)


def _poly_decay(lr, start_lr, end_lr, epochs, steps, decay_step, power):
    return (start_lr - end_lr) * (1 - steps / decay_step) ** power + end_lr


def _stair_decay(lr, start_lr, epochs, steps, decay_step, decay_rate):
    return start_lr * decay_rate ** (steps // decay_step)


def _eval_psnr(outputs, label, max_val, name, **kwargs):
    if not isinstance(outputs, list):
        outputs = [outputs]
    if isinstance(label, Image):
        label = img_to_array(label.convert('RGB'))
    for outp in outputs:
        if isinstance(outp, Image):
            outp = img_to_array(outp.convert('RGB'))
        label = np.squeeze(label)
        outp = np.squeeze(outp)
        mse = np.mean(np.square(outp - label))
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        print(f'{name}\'s PSNR = {psnr:.2f}dB')


def _image_align(image, scale, **kwargs):
    H, W = image.shape[-3:-1]
    h = H // scale * scale
    w = W // scale * scale
    return image[..., :h, :w, :]


def save_image(save_dir='.', output_index=-1, **kwargs):
    return partial(_save_model_predicted_images, save_dir=save_dir, index=output_index, **kwargs)


def print_psnr(max_val=255.0):
    return partial(_eval_psnr, max_val=max_val)


def reduce_residual(**kwargs):
    return partial(_sub_residual, **kwargs)


def to_rgb(**kwargs):
    return partial(_colored_grayscale_image, **kwargs)


def to_gray():
    def _gray_colored_image(inputs, **kwargs):
        return inputs[..., 0:1]

    return _gray_colored_image


def to_uv():
    def _uv_colored_image(inputs, **kwargs):
        return inputs[..., 1:]

    return _uv_colored_image


def add_noise(sigma, mean=0, clip=False):
    return partial(_add_noise, stddev=sigma, mean=mean, clip=clip)


def add_random_noise(low, high, step=1, mean=0, clip=False):
    return partial(_add_random_noise, low=low, high=high, step=step, mean=mean, clip=clip)


def lr_decay(method, lr, **kwargs):
    if method == 'exp':
        return partial(_exponential_decay, start_lr=lr, **kwargs)
    elif method == 'poly':
        return partial(_poly_decay, start_lr=lr, **kwargs)
    elif method == 'stair':
        return partial(_stair_decay, start_lr=lr, **kwargs)
    else:
        raise ValueError('invalid decay method!')


def blur(kernel_width, kernel_size, method='gaussian'):
    return partial(_gaussian_blur, width=kernel_width, size=kernel_size)


def image_alignment(scale):
    return partial(_image_align, scale=scale)
