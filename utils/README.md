# Install submodules
  - if you clone this repo with `--recurse-submodules` option, go to Setup
  - else
    - `git submodule init`
    - `git submodule update`

# Setup

## For Chamfer Distance
  - `cd ChamferDistancePytorch/chamfer3D`
  - `python setup.py install`
  - `git config submodule.utils/ChamferDistancePytorch.ignore all`
    - to ignore the new files when install
## For Earth Mover's Distance
  - `cd PyTorchEMD`  
  - `python setup.py install`
    - if `identifier "AT_CHECK" is undefined` error occured, fix the line 16, 17 in the file `cuda/emd_kernel.cu` such as `AT_CHECK` to `TORCH_CHECK`.
    