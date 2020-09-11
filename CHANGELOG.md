1.0.8

## 1.0.8
## 2020-09
- Refactor unit tests and tools

## 1.0.7
## 2020-08
- Fix loader error when inferring FRVSR
- Group common arguments
- Add RLSP

## 1.0.6
## 2020-07
- Update TF backend
- Add support to tensorflow 2.0 (both legacy and eager mode)
- Refactor torch backend models
- Add `--caching_dataset` to cache transformed data into memory (ignored when `memory_limit` set).
- Fix FastMetrics multi-threads issue
- Fix loading issue when inferring with VSR models

## 1.0.5
## 2020-05
- Fix bugs of DataLoader #108, #109

## 1.0.4
## 2020-04
- Fix an error that dataloader may mess up the file order
- Add a dataset checker to help verify the DDF
- Fix a bug of memory_limit [issue](https://github.com/LoSealL/VideoSuperResolution/issues/102)

## 1.0.3
## 2020-02
- Add --export to Train/eval.py

## 1.0.2
## 2020-02-16
- Fix wrong data format in ImageProcess
- Google Drive download api expired, fallback to a link message

## 1.0.1
## 2020-02-16
- Add SRMD (CVPR 2018) by Kai Zhang et. al.
- Upload to [PyPI](https://pypi.org/project/VSR/).

## 1.0.0
### 2020-01-31
- Trace changelog since 1.0.0
- Move VSRTorch package into VSR
    - Use `~/.vsr/config.yml` (`C:\Users\<user>\.vsr\config.yml` for **windows**) to config the package
    - Provide common API for users to add into their own project easily
- Refresh detailed [documents](./Docs)
