import torch
import torch.nn as nn

class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.log_cosh_loss = LogCoshLoss()
        self.l2_loss = nn.MSELoss()

        self.vgg_loss_coef = 0.4
        self.pixel_loss_coef = 1.5
        self.l1_penalty = 0.3
    
    def forward(
        self, 
        real_features, generated_features,
        real_image=None, generated_image=None, 
        average_dlatents=None, dlatents=None,
    ):           
        loss = 0
        # L1 loss on VGG16 features
        if self.vgg_loss_coef != 0:
            loss += self.vgg_loss_coef * self.l2_loss(real_features, generated_features)

        # + logcosh loss on image pixels
        if real_image is not None and generated_image is not None:
            loss += self.pixel_loss_coef * self.log_cosh_loss(real_image, generated_image)

        # Dlatent Loss - Forces latents to stay near the space the model uses for faces.
        if average_dlatents is not None and dlatents is not None:
            loss += self.l1_penalty * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        loss = true - pred
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))