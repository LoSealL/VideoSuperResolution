#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

from typing import Generator
from . import LOG


def to_list(x, repeat=1):
  """convert x to list object

    Args:
       x: any object to convert
       repeat: if x is to make as [x], repeat `repeat` elements in the list
  """
  if isinstance(x, (Generator, tuple, set)):
    return list(x)
  elif isinstance(x, list):
    return x
  elif isinstance(x, dict):
    return list(x.values())
  elif x is not None:
    return [x] * repeat
  else:
    return []


def str_to_bytes(s):
  """convert string to byte unit. Case insensitive.

  >>> str_to_bytes('2GB')
    2147483648
  >>> str_to_bytes('1kb')
    1024
  """
  s = s.replace(' ', '')
  if s[-1].isalpha() and s[-2].isalpha():
    _unit = s[-2:].upper()
    _num = s[:-2]
  elif s[-1].isalpha():
    _unit = s[-1].upper()
    _num = s[:-1]
  else:
    return float(s)
  if not _unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'):
    raise ValueError('invalid unit', _unit)
  carry = {'B': 1,
           'KB': 1024,
           'MB': 1024 ** 2,
           'GB': 1024 ** 3,
           'TB': 1024 ** 4,
           'PB': 1024 ** 5,
           'EB': 1024 ** 6,
           'ZB': 1024 ** 7,
           'YB': 1024 ** 8}
  return float(_num) * carry[_unit]


def cross_type_assign(value, dtype):
  """Convert `value` to `dtype`.
    Usually this can be done by simply `dtype(value)`, however, this ain't
    correct for str -> bool conversion.
  """

  if dtype is bool and isinstance(value, str):
    if value.lower() == 'false':
      return False
    elif value.lower() == 'true':
      return True
    else:
      LOG.warning(
        "suspect wrong typo {}, do you mean true/false?".format(value))
      return True
  return dtype(value)


def suppress_opt_by_args(opt, *args):
  """Use cmdline arguments to overwrite tf declared in yaml file.
    Account for safety, writing section not declared in yaml is not allowed.
  """

  def parse_args(argstr: str, prev_argstr: str):
    if prev_argstr:
      k, v = prev_argstr, argstr
    elif argstr.startswith('--'):
      if '=' in argstr:
        k, v = argstr[2:].split('=')
      else:
        k = argstr[2:]
        v = None
    elif argstr.startswith('-'):
      if '=' in argstr:
        k, v = argstr[1:].split('=')
      else:
        k = argstr[1:]
        v = None
    else:
      raise KeyError("Unknown parameter: {}".format(argstr))
    return k, v

  prev_arg = None
  for arg in args:
    key, value = parse_args(arg, prev_arg)
    prev_arg = None  # clear after use
    if key and value:
      # dict support
      keys = key.split('.')
      if keys[0] not in opt:
        raise KeyError("Parameter {} doesn't exist in model!".format(key))
      old_v = opt.get(keys[0])
      if isinstance(old_v, (list, tuple)):
        # list, tuple support
        if not value.startswith('[') and not value.startswith('('):
          raise TypeError("Invalid list syntax: {}".format(value))
        if not value.endswith(']') and not value.endswith(')'):
          raise TypeError("Invalid list syntax: {}".format(value))
        values = value[1:-1].split(',')
        if len(values) == 1 and values[0] == '':
          # empty list
          values = []
        new_v = [cross_type_assign(nv, type(ov)) for ov, nv in
                 zip(old_v, values)]
        opt[keys[0]] = new_v
      elif isinstance(old_v, dict):
        # dict support
        try:
          for k in keys[1:-1]:
            old_v = old_v[k]
          ref_v = old_v
          old_v = old_v[keys[-1]]
        except KeyError:
          raise KeyError("Parameter {} doesn't exist in model!".format(key))
        if isinstance(old_v, (list, tuple)):
          raise NotImplementedError("Don't support nested list type.")
        new_v = cross_type_assign(value, type(old_v))
        ref_v[keys[-1]] = new_v
      else:
        new_v = cross_type_assign(value, type(old_v))
        opt[keys[0]] = new_v
    elif key:
      prev_arg = key

  if prev_arg:
    raise KeyError("Parameter missing value: {}".format(prev_arg))
