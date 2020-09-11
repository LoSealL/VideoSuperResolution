"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-7

Visualize optical flow using color wheel
"""
import numpy as np

from ..Backend import DATA_FORMAT
from .ImageProcess import array_to_img


def _color_wheel():
    red_yellow, yellow_green, green_cyan = 15, 6, 4
    cyan_blue, blue_magenta, magenta_red = 11, 13, 6
    colors = [red_yellow, yellow_green, green_cyan,
              cyan_blue, blue_magenta, magenta_red]
    color = np.zeros([np.sum(colors), 3])
    for i in range(red_yellow):
        color[i] = [255, 255 * i / red_yellow, 0]
    for i in range(yellow_green):
        color[i + np.sum(colors[:1])] = [255 - 255 * i / yellow_green, 255, 0]
    for i in range(green_cyan):
        color[i + np.sum(colors[:2])] = [0, 255, 255 * i / green_cyan]
    for i in range(cyan_blue):
        color[i + np.sum(colors[:3])] = [0, 255 - 255 * i / cyan_blue, 255]
    for i in range(blue_magenta):
        color[i + np.sum(colors[:4])] = [255 * i / blue_magenta, 0, 255]
    for i in range(magenta_red):
        color[i + np.sum(colors[:5])] = [255, 0, 255 - 255 * i / magenta_red]
    return color / 255


def _viz_flow(u, v, logscale=True, scaledown=6):
    """
    Copied from @jswulff:
      https://github.com/jswulff/pcaflow/blob/master/pcaflow/utils/viz_flow.py

    top_left is zero, u is horizon, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    color_wheel = _color_wheel()
    n_cols = color_wheel.shape[0]

    radius = np.sqrt(u ** 2 + v ** 2)
    if logscale:
        radius = np.log(radius + 1)
    radius = radius / scaledown
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot + 1) / 2 * (n_cols - 1)  # -1~1 mapped to 0~n_cols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., n_cols

    k1 = k0 + 1
    k1[k1 == n_cols] = 0

    f = fk - k0

    n_colors = color_wheel.shape[1]
    img = np.zeros(u.shape + (n_colors,))
    for i in range(n_colors):
        tmp = color_wheel[:, i]
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


def visualize_flow(flow, v=None):
    if DATA_FORMAT == 'channels_last':
        u = flow[..., 0] if v is None else flow
        v = flow[..., 1] if v is None else v
    else:
        u = flow[0, :, :] if v is None else flow
        v = flow[1, :, :] if v is None else v
    viz = _viz_flow(u, v)
    return array_to_img(viz, 'RGB')
