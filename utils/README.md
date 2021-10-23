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
    - to ignore the new files

## For Earth Mover's Distance ([Link](https://github.com/Colin97/MSN-Point-Cloud-Completion))
  - `cd emd`  
  - `python setup.py install`
    