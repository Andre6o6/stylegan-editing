# Using GAN latent space for semantic photo editing
This repo contains the code for my masters thesis on face generation.
It explores the possibility of editing an image by projecting it into GAN latent space, then modifying its latent code
using some vector arithmetic, and finally generating modified image.

**To check it out, I highly recommend running this Colab notebook:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg\)](https://colab.research.google.com/github/Andre6o6/stylegan-editing/blob/master/StyleGAN_edit_images.ipynb) 

## Setup

In this work I use PyTorch implementation of StyleGAN ([from here](https://github.com/genforce/interfacegan)), but I'm planning to switch to PyTorch implementation of [StyleGAN2](https://github.com/NVlabs/stylegan2-ada).

Install dependances:
`
pip install facenet-pytorch
`

First, align images:
`
python align/align_images.py raw_images/ aligned_images/ --output_size=1024
`

Then, project images into latent space using latent optimization:
`
python encode_images.py
`

Lastly, perform feature transfer:
`
python edit_images.py --input path/to/input/latent --exemplar path/to/exemplar/latent
`

You can get help for command line arguments by typing `-h`:
`
python encode_images.py -h
python edit_images.py -h
`

## How it works
First step is to find corresponding latent code for the image. Since we can backpropagate gradients w.r.t. input vector 
throught generator network, we can directly optimize the latent code using some reconstruction loss function.
In this case it's a weighted sum of L2 loss in pixel space and L2 loss in feature space (that is, on some layer of pretrained VGG network).

![optimization](docs/optim_pipeline.gif)

Then we need to find a certain direction, corresponding to a change in desired attribute.
After that we can linearly shift the latent vector *g*(**x** + a**n**) to make this attribute more/less prominent.
