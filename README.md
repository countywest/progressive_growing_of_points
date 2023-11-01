# PGpoints
Pytorch implementation of the paper [Progressive Growing of Points with Tree-structured Generators](https://www.bmvc2021-virtualconference.com/assets/papers/0590.pdf) (BMVC 2021)

**Hyeontae Son, Young Min Kim**

<img src="./gifs/airplane.gif" width=400><img src="./gifs/car.gif" width=400>

```bash
@inproceedings{Son_2021_BMVC,
  author    = {Hyeontae Son and Young Min Kim},
  title     = {Progressive Growing of Points with Tree-structured Generators},
  booktitle = {32nd British Machine Vision Conference 2021, {BMVC} 2021, Online, November 22-25, 2021},
  pages     = {44},
  year      = {2021}
}
```

## Prerequisites

### Clone this repository
- `git clone --recurse-submodules https://github.com/countywest/progressive_growing_of_points`

### Install Dependencies
- `conda create -n pgpoints python=3.6` and `conda activate pgpoints`
- Install pytorch (1.4.0) & torchvision (0.5.0)
  - `pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html`
- Install other dependencies
  - `pip install -r requirements.txt`
- Install CD & EMD loss in pytorch ([Link](https://github.com/countywest/progressive_growing_of_points/tree/master/utils))

### Download datasets
- [ShapeNet](https://drive.google.com/file/d/1QLOYx6FCuqCILYCWJQ_443twCM7hbr3B/view?usp=sharing)
  - We downloaded the original dataset from [here](https://github.com/optas/latent_3d_points#data-set), and divided train/valid/test set with portion (85/5/10 % each)
    - train/valid/test list is provided in ```configs/shapenet_2048/*.list```
- [PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [TopNet](https://drive.google.com/file/d/1qDzvHX214pUiUbAmedQEH048rKo0WIEw/view?usp=sharing)
  - We downloaded the original dataset from [here](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip).
  - Since TopNet dataset does not provide the ground truth for test data, we used the provided validation set for testing and picked 600 samples from the training data to use it as a validation set.

### Make symlinks for the datasets
- ```mkdir data```
- ```ln -s [path to the dataset] data/[dataset name]```
  - dataset name: ```shapenet_2048, pcn_16384, topnet_2048```
  
## Usage
To train PGpoints,
```python train.py --model_type [MODEL_TYPE] --model_id [MODEL_ID]```
  - ```MODEL_TYPE``` should be one of ```auto_encoder, l-GAN, point_completion```
  - ```MODEL_ID``` should be exactly same as model id in the ```MODEL_TYPE.yaml```

To test PGpoints,
```python test.py --model_type [MODEL_TYPE] --model_id [MODEL_ID]```
  - this tests the best model in the ```logs/[MODEL_TYPE]/[MODEL_ID]```
  
  
## Acknowledgements
This project is influenced by following awesome works!
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation (ICLR 2018)](https://github.com/tkarras/progressive_growing_of_gans)
- [Multiresolution Tree Networks for 3D Point Cloud Processing (ECCV 2018)](https://github.com/matheusgadelha/MRTNet)
- [TopNet: Structural Point Cloud Decoder (CVPR 2019)](https://github.com/lynetcha/completion3d)
- [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shu_3D_Point_Cloud_Generative_Adversarial_Network_Based_on_Tree_Structured_ICCV_2019_paper.pdf)
- [A Progressive Conditional Generative Adversarial Network for Generating Dense and Colored 3D Point Clouds (3DV 2020)](https://github.com/robotic-vision-lab/Progressive-Conditional-Generative-Adversarial-Network)
