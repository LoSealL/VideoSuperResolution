#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 17

import numpy as np


class Ensembler:
  @staticmethod
  def expand(feature: np.ndarray):
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
  def merge(outputs: [np.ndarray]):
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
