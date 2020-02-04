#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
from functools import partial


def _exponential_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  return start_lr * decay_rate ** (steps / decay_step)


def _poly_decay(start_lr, end_lr, steps, decay_step, power, **kwargs):
  return (start_lr - end_lr) * (1 - steps / decay_step) ** power + end_lr


def _stair_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  return start_lr * decay_rate ** (steps // decay_step)


def _multistep_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  if not decay_step:
    return start_lr
  for n, s in enumerate(decay_step):
    if steps <= s:
      return start_lr * (decay_rate ** n)
  if steps > decay_step[-1]:
    return start_lr * (decay_rate ** len(decay_step))


def lr_decay(method, lr, **kwargs):
  if method == 'exp':
    return partial(_exponential_decay, start_lr=lr, **kwargs)
  elif method == 'poly':
    return partial(_poly_decay, start_lr=lr, **kwargs)
  elif method == 'stair':
    return partial(_stair_decay, start_lr=lr, **kwargs)
  elif method == 'multistep':
    return partial(_multistep_decay, start_lr=lr, **kwargs)
  else:
    print('invalid decay method!')
    return None
