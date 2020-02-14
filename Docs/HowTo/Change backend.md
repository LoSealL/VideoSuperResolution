# How to change VSR backend
The VSR uses PyTorch as the default backend. And VSR also supports tensorflow
for some of models.

Edit config file `~/.vsr/config.yml`, If you'd like to change to tensorflow:
(create one if not exist)
```yaml
# the backend could be 'tensorflow', 'tensorflow2', 'pytorch'
backend: tensorflow
# the verbose could be 'error', 'warning', 'info', 'debug'
verbose: info
```
