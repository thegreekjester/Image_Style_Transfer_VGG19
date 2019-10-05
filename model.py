import torch 
from torchvision import models


def getModel():
    """
    Returns the feature extracting parts of the VGG19 model
    """
    # get only the non-classifer (non FCL) section of the vgg network 
    vgg = models.vgg19(pretrained=True).features

    # freeze all the layers so that we can utilize the network unchanged
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    return vgg 