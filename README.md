# Progressive Growing of Points
![airplane](./teaser/airplane.gif)
![chair](./teaser/chair.gif)

## Prerequisites
#### Clone this repository
```git clone --recurse-submodules https://github.com/countywest/progressive_growing_of_points.git```

#### Download & Link datasets
  - ShapeNet Data from [this](https://github.com/optas/latent_3d_points)
  - split dataset to train/valid/test
  - move dataset to ```progressive_growing_of_data/data/shapenet_2048```

#### Install Dependencies
```pip install -r requirements.txt```

## Usage
  - ```python train.py``` for training
  - ```python test.py``` for testing