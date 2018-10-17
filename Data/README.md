## DDF
Dataset Description Format

===changelog===
- 2018.10.17 Mitigate DDF from json to yaml
- 2018.08.21 Add "Path_Tracked" section to alias test set
 
#### Path
In `Path` section, a file glob pattern is bound to a name, the pattern is from `pathlib.Path.glob`, you can find
more information [here](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob).

#### Path_Tracked
`Path_Tracked` section is almost the same as `Path`. It is recently added to automatically add these paths to
individual test dataset. Paths under this section is usually test-only data.

#### Dataset
In `Dataset` section, we bind a number of **Path** to a dataset name.
In each dataset, you can specify a `train` set, a `val` set a `test` set and an `infer` set. Every set can point
to a single path of a path array, or even a pure file glob pattern.
For RAW data (such as RGB, YUV et.al.) you should explicitly specify `mode`, `width` and `height` (see `MCL-V` for example).

## Note
[dataset.json](datasets.json) has been deprecated now, but is kept for compatibility.
Please use [dataset.yaml](datasets.yaml) instead.