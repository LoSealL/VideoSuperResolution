## Dataset JSON
I use a json file to describe datasets, avoid including huge image files in git repository

## The pattern in JSON
The JSON contains **TWO** main descriptors: `Path` and `Dataset`
- `Path` is a collection of **name** and local **url** of data files, the **url** satisfies the `Path.glob` patterns
- `Dataset` is a set of collections combines multiple collections defined in `Path` to form a **dataset** which has `train`, `val`, and `test` offerings. The value of `train`, `val` and `test` can be either a list of names defined in `Path` or a pure url pattern. The collection also can optionaly have a `param` entry to pass extra attributes to dataset. For instance the height and width parameters to a raw dataset.

---update 8.21---

- Add a new entry `Path_Tracked`, in which values are also added to `Dataset` as a **test** set

### *Note
If **url** in `Path` points to a directory, then dataset will scan that directory and store each file and child-directory into nodes.
If the node is a directory, it then is treated as a container of a sequence of images, and returns image in that directory one by one in the order of file name.

You can check my `datasets.json` as an example.

```json
{
  "Path": {
    name: url
  },
  "Path_Tracked": {
    name: url
  },
  "Dataset": {
    dataset-name: {
      "train": [train1, train2, ...],
      "val": [val1, val2, ...]
      "test": [test1, test2, ...]
      // optional for raw data, don't need for compressed data formats
      "param": {
        "mode": "YV12", // NV12, NV21, YV12, YV21, RGBA, BGRA is recongnized for now
        "width": 1920,
        "height": 1080,
      }
    }
  }
}
```
