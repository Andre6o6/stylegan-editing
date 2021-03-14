import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from interfacegan.models.stylegan_generator import StyleGANGenerator
from models.latent_optimization import denormalize, LatentOptimizer
from models.initial_prediction import InitialLatentPredictor
from models.loss import LatentLoss
from utils.blur import GaussianSmoothing

# Get target image tensor and target features
def prepare_reference_image(imagepath):
    reference_image = transforms.ToTensor()(Image.open(imagepath)).unsqueeze(0).to(device)
    with torch.no_grad():
        prep_image = latent_optimizer.preprocess(reference_image)
        reference_features = latent_optimizer.feature_extractor(prep_image)
    return reference_image, reference_features

# Get initial prediction for latent
def prepare_latents(latent_predictor_path=None, latent_space="WP"):
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
             n_iters=100, cascade=True, smoothing=None):
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
        if cascade and step < 5*5:
            k_map = 3 + 2*(step//5)
            initial_latents.grad[0, k_map:].zero_()
        optimizer.step()
        progress_bar.set_description("Loss = {}".format(loss.item()))

    optimized_dlatents = initial_latents.detach().cpu().numpy()     #FIXME mb return tensors
    return optimized_dlatents

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dlatent_path = "latents/"
latent_predictor_path = "image_to_latent.pt"
aligned_images = "aligned_images/"

# StyleGAN is converted from official tensorflow model to pytorch
converted_model = StyleGANGenerator("stylegan_ffhq")
avg_latents = converted_model.model.truncation.w_avg.view((-1, 1, 512))

# Initialize latent optimization pipeline (generator > vgg)
latent_optimizer = LatentOptimizer(converted_model.model, vgg_layer=12, latent_space="WP")

smoothing = GaussianSmoothing(3, kernel_size=7, sigma=5).to(device)

for file in os.listdir(aligned_images):
    print(file)
    if file.split('.')[-1] not in ["png", "jpg", "jpeg"]:
        continue

    # Load target image and get features
    image_path = aligned_images + file
    reference_image, reference_features = prepare_reference_image(image_path)
    
    # Predict initial latent vector or randomly sample it
    initial_latents = prepare_latents(latent_predictor_path, latent_optimizer.latent_space)

    optimized_dlatents = optimize(reference_image, reference_features, initial_latents, \
             n_iters=200, cascade=True, smoothing=smoothing)
    
    # Save optimized latent vector
    np.save(dlatent_path+image_path.split("/")[-1]+".npy", optimized_dlatents)
