"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-4-17

Check the validation of a dataset name
"""
import argparse
from pathlib import Path

from VSR.DataLoader import load_datasets


def main(entry: str, ddf_file):
    entry = entry.upper()
    all_data = load_datasets(ddf_file)
    if entry not in all_data:
        raise KeyError(f"The dataset `{entry}` not found in the DDF")

    data = all_data.get(entry)
    print(f"Dataset: {data.name}")

    def _check(name: str):
        print(f"\n=========  CHECKING  {name}  =========\n")
        if name in data and data[name] is not None:
            print(f"Found `{name}` set in \"{data.name}\":")
            _hr = data[name].hr
            _lr = data[name].lr
            video_type = _hr.as_video
            if video_type:
                print(f"\"{data.name}\" is video data")
            if _hr is not None:
                _hr = _hr.compile()
                print(f"Found {len(_hr)} ground-truth {name} data")
            if _lr is not None:
                _lr = _lr.compile()
                print(f"Found {len(_lr)} custom degraded {name} data")
                if len(_hr) != len(_lr):
                    print(
                        f" [E] Ground-truth data and degraded data quantity not matched!!")
                elif video_type:
                    for x, y in zip(_hr, _lr):
                        if x.frames != y.frames:
                            print(
                                f" [E] Video clip {x.name}|{y.name} quantity not matched!!")
        else:
            print(f"{data.name} doesn't contain any {name} data.")

    _check('train')
    _check('val')
    _check('test')


if __name__ == '__main__':
    CWD = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Check the dataset and print out its content")
    parser.add_argument("dataset", type=str,
                        help="The name of the dataset, case insensitive.")
    parser.add_argument("--description-file", default=f"{CWD}/Data/datasets.yaml",
                        help="DDF file")
    flags = parser.parse_args()
    main(flags.dataset, flags.description_file)
