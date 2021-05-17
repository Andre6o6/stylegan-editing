"""This module implements Latent Optimization, aimed to project input images into
StyleGAN's latent space.
"""
import argparse
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from interfacegan.models.stylegan_generator import StyleGANGenerator
from models.latent_optimization import denormalize, LatentOptimizer
from models.initial_prediction import InitialLatentPredictor
from models.loss import LatentLoss
from utils.blur import GaussianSmoothing


def prepare_reference_image(imagepath):
    """Get target image tensor and target features tensor."""
    reference_image = transforms.ToTensor()(Image.open(imagepath)).unsqueeze(0).to(device)
    with torch.no_grad():
        prep_image = latent_optimizer.preprocess(reference_image)
        reference_features = latent_optimizer.feature_extractor(prep_image)
    return reference_image, reference_features


def prepare_latents(latent_predictor_path=None, latent_space="WP"):
    """Get initial prediction for latent vector."""
    if latent_predictor_path and latent_space == "WP":
        image_to_latent = InitialLatentPredictor().to(device)
        image_to_latent.load_state_dict(torch.load(latent_predictor_path))
        image_to_latent.eval()

        with torch.no_grad():
            initial_latents = image_to_latent(prep_image)
        initial_latents = initial_latents.detach().to(device).requires_grad_(True)
    elif latent_optimizer.latent_space == "WP":
        initial_latents = torch.zeros((1,18,512)).to(device).requires_grad_(True)
    else:
        initial_latents = torch.zeros((1,512)).to(device).requires_grad_(True)
    return initial_latents


def optimize(reference_image, reference_features, initial_latents, 
             n_iters=100, cascade=True, cascade_iters=50, smoothing=None):
    """Perform latent optimization, only on crude feature maps first few steps."""
    # Loss and optimizer
    criterion = LatentLoss()
    optimizer = torch.optim.Adam([initial_latents], lr=0.025)
    
    # Iteratevly update latent vector using backprop
    progress_bar = tqdm(range(n_iters))
    for step in progress_bar:
        optimizer.zero_grad()

        generated_image_features, generated_image = latent_optimizer(initial_latents)
        
        if smoothing:   # blur generated image
            generated_image = F.pad(generated_image, (3,3,3,3), mode="reflect")
            generated_image = smoothing(generated_image)

        loss = criterion(
            reference_features, generated_image_features,
            reference_image, generated_image,
            #avg_latents, initial_latents
        )
        loss.backward()

        #Optimize crude features first, then finer: zero gradients of some maps
        if cascade and step < cascade_iters:
            initial_latents.grad[0, 5:].zero_()

        optimizer.step()

        if cascade and step < cascade_iters:
            with torch.no_grad():
                initial_latents[0, 5:] = initial_latents[0, :5].mean(dim=0)

        progress_bar.set_description("Loss = {}".format(loss.item()))

    optimized_dlatents = initial_latents.detach().cpu().numpy()     #FIXME mb return tensors
    return optimized_dlatents


def arg_parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Latent Optimization module")
    parser.add_argument(
        "--root",
        dest="root_dir",
        help="Aligned images directory",
        default="aligned_images/",
        type=str,
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        help="Latent vectors directory",
        default="latents/",
        type=str,
    )
    parser.add_argument(
        "--predictor",
        dest="predictor_path",
        help="Initial predictor path",
        default="image_to_latent.pt",
        type=str,
    )
    parser.add_argument(
        "--iters",
        dest="n_iters",
        help="Number of iterations",
        default=400,
        type=int,
    )
    parser.add_argument(
        "--cascade",
        dest="cascade_iters",
        help="Number of first iterations, where only crude feature maps are optimized",
        default=50,
        type=int,
    )
    return parser.parse_args()


def main():
    args = arg_parse()
    aligned_images = args.root_dir
    dlatent_path = args.out_dir
    use_cascade = args.cascade_iters > 0
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # StyleGAN is converted from official tensorflow model to pytorch
    print("Loading StyleGAN, pretrained model should be in 'interfacegan/models/pretrain/'")
    converted_model = StyleGANGenerator("stylegan_ffhq")

    # Initialize latent optimization pipeline (generator -> vgg)
    latent_optimizer = LatentOptimizer(converted_model.model, vgg_layer=12, latent_space="WP")

    smoothing = GaussianSmoothing(3, kernel_size=7, sigma=5).to(device)
    
    print("Starting latent optimization...")
    for file in os.listdir(aligned_images):
        print(file)
        if file.split('.')[-1] not in ["png", "jpg", "jpeg"]:
            continue

        # Load target image and get features
        image_path = os.path.join(aligned_images, file)
        reference_image, reference_features = prepare_reference_image(image_path)
        
        # Predict initial latent vector or randomly sample it
        initial_latents = prepare_latents(args.predictor_path, latent_optimizer.latent_space)

        optimized_dlatents = optimize(reference_image, reference_features, initial_latents, \
            n_iters=args.n_iters, cascade=use_cascade, cascade_iters=args.cascade_iters, smoothing=smoothing)
        
        # Save optimized latent vector
        dlatent_name = image_path.split("/")[-1]+".npy"
        np.save(os.path.join(dlatent_path, dlatent_name), optimized_dlatents)


if __name__ == "__main__":
    main()
