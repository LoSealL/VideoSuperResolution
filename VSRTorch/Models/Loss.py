#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/9 下午2:41

import torch
import torch.nn.functional as F


def total_variance(x, dims=(2, 3), reduction='mean'):
  tot_var = 0
  reduce = 1
  for dim in dims:
    row = x.split(1, dim=dim)
    reduce *= x.shape[dim]
    for i in range(len(row) - 1):
      tot_var += torch.abs(row[i] - row[i + 1]).sum()
  if reduction != 'mean':
    reduce = 1
  return tot_var / reduce
