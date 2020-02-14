# Add Models to VSR
version 0.0.1-alpha

## Write new model with VSR

- Create a new python file in VSR.Models.
- Copy codes from `Edsr.py` and rename the class `EDSR`.
- Write model graph in function `build_graph`
- Write loss in function `build_loss` (You can also write in `build_graph` and ignore `build_loss`, that's your choice.)
- (Optional) Write summaries and savers. (There're default summary and saver)

## Register model into VSR

- Open `VSR.Models.__init__.py`.
- Add an entry to `models` dict. The entry is `alias: (file-name, class-name)`.

----

## Write new model with VSRTorch

- Create a new python file in VSRTorch.Models.
- Copy codes from `Espcn.py` and rename the class `ESPCN`.
- Write modules.
- Write forward and backward data-path in function `train`.
- Write forward data-path in function `eval`.

## Register model into VSRTorch

- Open `VSRTorch.Models.__init__.py`.
- Add an entry to `models` dict. The entry is `alias: (file-name, class-name)`.