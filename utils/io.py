import numpy as np
import cv2
from PIL import Image

def load_image(path):
    image = np.asarray(Image.open(path))
    image = np.transpose(image, (2,0,1))  #WxHxC to CxWxH
    return image

def save_image(image, path):
    image = np.transpose(image, (1,2,0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

def get_image_numpy(tensor, saturate=False):
    img = tensor.detach().cpu().squeeze().numpy()
    img = img.transpose((1,2,0))
    if saturate:
        img = (img - img.min())/(img.max() - img.min())
    img = (255*img).astype(np.uint8)
    return img
