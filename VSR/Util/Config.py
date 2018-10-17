"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Parse configuration from multiple sources.
Support:
1. JSON, YAML
2. dict, easydict
"""

import easydict
import yaml


class Config(easydict.EasyDict):
    def __init__(self, obj=None, **kwargs):
        super(Config, self).__init__(**kwargs)
        # delete method name in map
        try:
            self.pop('update')
            self.pop('pop')
        except AttributeError:
            pass
        if obj is not None:
            assert isinstance(obj, (dict, str))
            if isinstance(obj, str):
                with open(obj, 'r') as fd:
                    obj = yaml.load(fd)
            self.update(**obj)

    def __getattr__(self, item):
        return self.get(item)

    def update(self, E=None, **F):
        # Fix update error of easydict
        d = E or dict()
        d.update(F)
        for k in d:
            self.__setattr__(k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)
