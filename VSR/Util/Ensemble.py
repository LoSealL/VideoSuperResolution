"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-6-17

Ensemble helper
"""
from typing import List

import numpy as np


class Ensembler:
    """Ensemble image by rotate, flip then average the corresponded outputs
    """

    @staticmethod
    def expand(feature: np.ndarray):
        """Extend image to 8 different angles
        """

        r0 = feature.copy()
        r1 = np.rot90(feature, 1, axes=[-3, -2])
        r2 = np.rot90(feature, 2, axes=[-3, -2])
        r3 = np.rot90(feature, 3, axes=[-3, -2])
        r4 = np.flip(feature, axis=-2)
        r5 = np.rot90(r4, 1, axes=[-3, -2])
        r6 = np.rot90(r4, 2, axes=[-3, -2])
        r7 = np.rot90(r4, 3, axes=[-3, -2])
        return r0, r1, r2, r3, r4, r5, r6, r7

    @staticmethod
    def merge(outputs: List[np.ndarray]):
        """Average all output images, see `Ensembler.expand`
        """

        results = []
        for i in outputs:
            outputs_ensemble = [
                i[0],
                np.rot90(i[1], 3, axes=[-3, -2]),
                np.rot90(i[2], 2, axes=[-3, -2]),
                np.rot90(i[3], 1, axes=[-3, -2]),
                np.flip(i[4], axis=-2),
                np.flip(np.rot90(i[5], 3, axes=[-3, -2]), axis=-2),
                np.flip(np.rot90(i[6], 2, axes=[-3, -2]), axis=-2),
                np.flip(np.rot90(i[7], 1, axes=[-3, -2]), axis=-2),
            ]
            results.append(
                np.concatenate(outputs_ensemble).mean(axis=0, keepdims=True))
        return results
