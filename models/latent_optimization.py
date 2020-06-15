import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

def denormalize(synthesized_image, min_value=-1, max_value=1):
    #Cast from [-1, 1] to [0, 255]; gradients should be ok ###(?)###
    synthesized_image = 255. * (synthesized_image - min_value) / (max_value - min_value)
    synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)
    return synthesized_image

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_layer=12):
        super().__init__()
        self.image_size = 256
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

        self.vgg16 = vgg16(pretrained=True).features[:vgg_layer].to(device).eval()

    def forward(self, image):
        image = image / 255.
        image = F.adaptive_avg_pool2d(image, self.image_size)
        image = (image - self.mean) / self.std
        features = self.vgg16(image)
        return features

class LatentOptimizer(nn.Module):
    def __init__(self, synthesizer, vgg_layer=9):
        super().__init__()
        self.synthesizer = synthesizer.to(device).eval()
        self.feature_extractor = VGGFeatureExtractor(vgg_layer)

    def forward(self, dlatents):
        generated_image = self.synthesizer(dlatents)
        generated_image = denormalize(generated_image)
        features = self.feature_extractor(generated_image)
        return features, generated_image