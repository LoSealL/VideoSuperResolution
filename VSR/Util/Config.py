"""
Copyright: Wenyi Tang 2017-2018
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
        if obj is not None:
            assert isinstance(obj, (dict, str))
            if isinstance(obj, str):
                with open(obj, 'r') as fd:
                    obj = yaml.load(fd)
            self.update(**obj)

    def __getattr__(self, item):
        return self.get(item)
