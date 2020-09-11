"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2018-8-1
"""
import importlib
import re
from pathlib import Path

__all__ = ['get_model', 'list_supported_models']


def auto_search(root):
    def _parse_class(file):
        obj = []
        key_to_remove = set()
        file = Path(file)
        with file.open('r', encoding='utf-8') as fd:
            line = fd.readline()
            while line:
                if line.startswith('class'):
                    # pylint: disable=anomalous-backslash-in-string
                    if '(SuperResolution)' in line:
                        try:
                            classname = re.compile(
                                "(?<=class\s)\w+\\b").findall(line)[0]
                            obj.append(classname)
                        except IndexError:
                            print(" [!] class: " + line)
                    else:
                        for cls in obj:
                            if f'({cls})' in line:
                                try:
                                    classname = re.compile(
                                        "(?<=class\s)\w+\\b").findall(line)[0]
                                    obj.append(classname)
                                    key_to_remove.add(cls)
                                except IndexError:
                                    print(" [!] class: " + line)
                line = fd.readline()
        for key in key_to_remove:
            obj.remove(key)
        return {file.stem: obj}

    mods = sorted(filter(
        lambda x: x.is_file() and not x.stem.startswith('__'),
        Path(root).glob('*.py')))
    for _m in mods:
        cls = _parse_class(_m)
        for k in cls:
            if k.lower() in models:
                print(" [!] duplicated model names found: " + k)
                continue
            if len(cls[k]) == 1:
                models[k.lower()] = (k, cls[k][0])
            elif len(cls[k]) > 1:
                for i in cls[k]:
                    models[f'{k.lower()}.{i.lower()}'] = (k, i)


models = {}
auto_search(Path(__file__).parent)


def get_model(name):
    module = f'.Backend.TF.Models.{models[name][0]}'
    package = 'VSR'
    m = importlib.import_module(module, package)
    return m.__dict__[models[name][1]]


def list_supported_models():
    return models.keys()
