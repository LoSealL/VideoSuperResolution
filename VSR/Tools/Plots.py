"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 28th 2018

Plot csv record files, a simple tool
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("url", type=str, help="csv file")
parser.add_argument("--savedir", type=str, default='/tmp/vsr/')
parser.add_argument("--x_offset", type=int, default=0)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--debug", type=bool, default=False)
args = parser.parse_args()

_DEBUG = args.debug


def _read_csv(url):
    with open(url, 'r') as fd:
        reader = csv.reader(fd)
        data = []
        fields = next(reader)
        for row in reader:
            data.append(np.asarray(row))
        return np.stack(data), fields


def _endless_shape(ls='-'):
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    markers = ['+', '^', 's', 'o', '1', '2', '3', '4', 'v', 'x', '*', 'D', 'p']
    for i in range(999999):
        yield dict(ls=ls,
                   c=colors[i % len(colors)],
                   marker=markers[i % len(markers)])


def gan_sweep_benchmark(title):
    data, fields = _read_csv(args.url)
    assert len(fields) == 4
    data = data.reshape([-1, 5, 4])
    p1_data = data[:6]
    p2_data = data[6:16]
    p3_data = data[16:]
    assert p2_data.shape[0] == p3_data.shape[0]
    X_AXIS = ['20k', '40k', '60k', '80k', '100k']

    def draw(m, caption, save):
        fig, _ = plt.subplots(2, 1, dpi=150, figsize=(10, 5))
        shape1 = _endless_shape('-')
        shape2 = _endless_shape('--')
        plt.subplot(211)
        plt.title(caption)
        plt.ylabel('FID')
        plt.ylim(20, 100)
        plt.subplot(212)
        plt.xlabel('iter')
        plt.ylabel('Inception score')
        plt.ylim(1, 8)
        for d in m:
            d0 = d[:, 0].astype('float')
            d1 = d[:, 1].astype('float')
            name = d[:, 2][0]
            plt.subplot(211)
            plt.plot(X_AXIS, d0, **next(shape1), label=name)
            plt.legend(fontsize='small',
                       loc="upper left",
                       handlelength=0,
                       borderpad=1,
                       bbox_to_anchor=(1, 0, 0, 0.5))
            plt.subplot(212)
            plt.plot(X_AXIS, d1, **next(shape2), label=name)
        if _DEBUG:
            plt.show()
        else:
            plt.savefig(args.savedir + '/' + save)

    draw(p1_data, title + ', D BN', title + '_bn.png')
    draw(p2_data, title + ', D w/o norm.', title + '_none.png')
    draw(p3_data, title + ', D SN', title + '_sn.png')


if __name__ == '__main__':
    fn = list(globals().keys())
    for k in fn:
        if k.startswith('_') or k in ('parser', 'args'):
            continue
        if args.task in k:
            if callable(globals()[k]):
                globals()[k](args.title)
