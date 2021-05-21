# CAR dataset

CAR is a SOTA image super-resolution technique on Set5 - 4x upscaling.
We adapted CAR technique to DL20 image dataset.

For more information, please go to the link https://github.com/sunwj/CAR.git

## Installation

```sh
# get CAR-pytorch source
git clone https://github.com/sunwj/CAR.git
cd CAR

# compile the code of the resampler
cd adaptive_gridsampler
python3 setup.py build_ext --inplace 
```

## Pre-trained models

You can download the pre-trained models for 2x and 4x downscaling and super-resolution from [here].

## Implement

The default code 4x downsamples the original image, and then 4x upsamples to the original image size using CAR algorithm.
We want to get 4x upsampled image, so we 4x upsample the images **(upsample.py)** and then implement the default code.

```sh
python3 run.py --scale 4 --img_dir path_to_images --model_dir path_to_pretrained_models
--output_dir path_to_output
```

[//]:
    [here]: <https://mega.nz/file/XzIm3YhT#jbIOOOGBOiKtv3VAOD782Mz7nK1L_kma-BzR-RhboW4>