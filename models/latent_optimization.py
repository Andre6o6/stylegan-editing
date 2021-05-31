import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

def denormalize(synthesized_image, min_value=-1, max_value=1):
    """Convert from range [min_value, max_value] to [0.0, 1.0]"""
    synthesized_image = (synthesized_image - min_value) / (max_value - min_value)
    synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=1)
    return synthesized_image

class VGGPrerocess(nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1,1,1)

    def forward(self, image):
        image = F.adaptive_avg_pool2d(image, self.image_size)
        image = (image - self.mean) / self.std
        return image

class LatentOptimizer(nn.Module):
    def __init__(self, model, vgg_layer=12, latent_space='WP'):
        super().__init__()
        
        assert latent_space in ['Z', 'W', 'WP'], "Only Z, W, WP are supported"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = model.to(device).eval()
        self.latent_space = latent_space
        self.preprocess = VGGPrerocess()
        self.feature_extractor = vgg16(pretrained=True).features[:vgg_layer].to(device).eval()

    def forward(self, dlatents):
        if self.latent_space == 'Z':
            zs = dlatents
            ws = self.model.mapping(zs)
            wps = self.model.truncation(ws)
        elif self.latent_space == 'W':
            ws = dlatents
            wps = self.model.truncation(ws)
        elif self.latent_space == 'WP':
            wps = dlatents

        generated_image = self.model.synthesis(wps)
        generated_image = denormalize(generated_image)
        preprocessed_image = self.preprocess(generated_image)
        features = self.feature_extractor(preprocessed_image)
        return features, generated_image
