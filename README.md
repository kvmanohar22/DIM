# Deep Image Matting

This is Chainer implementation of the paper **Deep Image Matting**

<p align="center"><img width="80%" height="50%" src="imgs/network.png"/></p>

## Example Results

## Installation
It's recommended to use anaconda environment.

### Requirements
- Linux
- Python 3
- chainer (4.3.1)
- cupy (4.3.0)
- scikit-image (0.13.0)

1. Install **chainer** and **cupy** from [here](https://docs.chainer.org/en/stable/install.html)
2. Install the other requirements using `pip`

## Usage

```bash
git clone https://github.com/kvmanohar22/DIM.git
cd DIM
```
Append root of this directory to `PYTHONPATH` environment variable

```export PYTHONPATH=$PYTHONPATH:`pwd`/..```

### Demo
`bash ./scripts/demo.sh`

The above script gives you a demo of images in `img` directory.

If you want to test on a custom image, then run the following:
`bash ./scripts/demo.sh path_to_image path_to_trimap`

The model is tested on **Image Matting Dataset**. Please contact Brian Price (bprice@adobe.com) for the dataset. 

### Train
If you have the above dataset:
`bash ./scripts/train.sh path_to_root_of_train_dataset`

To set more options, check out the `options.py` file and set them accordingly in `scripts/train.sh`
