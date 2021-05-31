"""This module implements Feature transfer using latent vectors from latent 
optimization.
"""
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1
from interfacegan.models.stylegan_generator import StyleGANGenerator
from models.latent_optimization import LatentOptimizer


def morph_coefficient(w_input, w_exemplar, boundary, map_k=None):
    if map_k is None:
        proj_x = np.inner(w_input[:, 0], boundary)
        proj_y = np.inner(w_exemplar[:, 0], boundary)
    else:
        proj_x = np.inner(np.mean(w_input[:, :map_k], axis=1), boundary)
        proj_y = np.inner(np.mean(w_exemplar[:, :map_k], axis=1), boundary)
    return proj_y[0,0] - proj_x[0,0]


def feature_morph(latent, boundary, effect_coef, latent_optimizer, facenet,
                  n_iters=100, identity_correction=True, identity_coef=2):
    
    device = latent_optimizer.device
    latent = torch.tensor(latent, dtype=torch.float32, device=device)
    latent.requires_grad_(True)

    with torch.no_grad():
        original_img = latent_optimizer.model.synthesis(latent)
        original_emb = facenet(original_img)

    boundary = torch.tensor(boundary, dtype=torch.float32, device=device)
    boundary_expended = torch.tile(boundary, (1,18,1)).to(device)

    path = []
    for step in range(n_iters):
        with torch.no_grad():
            latent += boundary * (effect_coef/n_iters)
        
        if identity_correction:
            if (latent.grad is not None):
                latent.grad.zero_()
            
            new_img = latent_optimizer.model.synthesis(latent)
            new_emb = facenet(new_img)
            loss = torch.inner(original_emb, new_emb)
            loss.backward()

            with torch.no_grad():
                gradients = latent.grad
                
                projection = torch.inner(gradients.squeeze(), boundary.squeeze())
                projected_b = projection.view(-1,1)*boundary_expended.squeeze()
                ord_gradients = gradients - projected_b

                latent += identity_coef * ord_gradients
        path.append(latent.detach().cpu().numpy())
    
    result = path[-1]
    return result, path


def arg_parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Latent Optimization module")
    parser.add_argument(
        "--input",
        dest="w_input",
        help="Path to input image latent vector",
        type=str,
    )
    parser.add_argument(
        "--exemplar",
        dest="w_exemplar",
        help="Path to exemplar image latent vector",
        type=str,
    )
    parser.add_argument(
        "--boundary",
        dest="boundary",
        help="One of 'smile', 'pose' or path to boundary vector",
        default="pose",
        type=str,
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        help="Results directory",
        default="out/",
        type=str,
    )
    parser.add_argument(
        "--iters",
        dest="n_iters",
        help="Number of iterations",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--identity",
        dest="identity",
        help="Identity correction coefficient",
        default=2.0,
        type=float,
    )
    return parser.parse_args()   


def main():
    args = arg_parse()
    identity_correction = args.identity > 0
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load latent vectors and boundary vector
    w_input = np.load(args.w_input)
    w_exemplar = np.load(args.w_exemplar)
    
    if (args.boundary.lower() == "smile"):
        boundary = np.load("boundaries/stylegan_ffhq_smile_w_boundary.npy")
    elif (args.boundary.lower() == "pose"):
        boundary = np.load("boundaries/stylegan_ffhq_pose_w_boundary.npy")
    else:
        boundary = np.load(args.boundary)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load GAN generator
    print("Loading StyleGAN, pretrained model should be in 'interfacegan/models/pretrain/'")
    converted_model = StyleGANGenerator("stylegan_ffhq")
    latent_optimizer = LatentOptimizer(converted_model.model, latent_space="WP")
    
    facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    
    # Feature transfer
    effect_coef = morph_coefficient(w_input, w_exemplar, boundary, map_k=5)
    result_latent, _ = feature_morph(w_input, boundary, effect_coef, latent_optimizer, \
        n_iters=args.n_iters, identity_correction=identity_correction, identity_coef=args.identity)
    
    # Save results
    latent_name = args.w_input.split('/')[-1]
    np.save(os.path.join(args.out_dir, latent_name), result_latent)
    
    generated_image = converted_model.synthesize(result_latent, "WP")["image"]
    generated_image = converted_model.postprocess(generated_image)[0]
    img_pil = Image.fromarray(generated_image)
    img_name = latent_name.rstrip(".npy")
    img_pil.save(os.path.join(args.out_dir, img_name))
    

if __name__ == "__main__":
    main()
