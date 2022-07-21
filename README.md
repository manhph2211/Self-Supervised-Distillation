Distillation with no labels :smile:
=====

# Introduction

In this repo, I want to implenment DINO with simple scenario that can be easily used to train and evaluate as well as observe the attention map of an image. 

![image](https://user-images.githubusercontent.com/61444616/180137286-6c7551a8-df35-4bd1-bacd-3b54d4618ffb.png)

DINO is such an impressive model that utilizes the superpower of both self-supervised learning and distillation applying for vision transformer models. Some of main points of [the paper](https://arxiv.org/abs/2104.14294):

- There is NO supervision at all. This is an self-supervised learning task which is quite similar with contrastive learning except some small modifications. Loss function here is a standard cross-entropy loss.

- When we deal with knowledge distillation, usually we mimic the output of a larger model as the teacher to compress the student. We can use both soft-label and hard-label to combine loss terms and enjoy the benefit of semi-supervised learning. Well, here DINO does not need labels and the teacher and student have the same network but different weights, sharing through Expotential Moving Avarage. 

I strongly refered to the [official implementation of DINO](https://github.com/facebookresearch/dino) so many thanks to the authors for contributing a great project.

# Usage 

## Dataset

Here I used the script to download the tinyimagenet dataset, you can simply just run:

```
cd data
bash download_data.sh
```

## Environment Setup

Conda should be used in this stage to create new environment and install some required packages:

```
conda create -n dino python=3.9
conda activate dino
conda install --file requirements.txt
export PYTHONPATH="path_to_repo/DINOMAX"
```

## Tools

You might wannt train with:

`python -m torch.distributed.launch --nproc_per_node=1 /tools/train.py --arch vit_small --data_path data/tiny-imagenet-200/train --output_dir /output`

and try this for generating attention of a given image:

`python /tools/visualize_attention.py --image_path data/tiny-imagenet-200/train/n02909870/images/n02909870_100.JPEG --pretrained_weights /output/checkpoint.pth`

# Results

# Reference

```
@misc{https://doi.org/10.48550/arxiv.2104.14294,
  doi = {10.48550/ARXIV.2104.14294},
  
  url = {https://arxiv.org/abs/2104.14294},
  
  author = {Caron, Mathilde and Touvron, Hugo and Misra, Ishan and Jégou, Hervé and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Emerging Properties in Self-Supervised Vision Transformers},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
