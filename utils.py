import torch 
import torchvision 
from torchvision import transforms
import io 
import PIL.Image
import requests
import numpy as np

def load_image(img, max_size=400, shape=None):

    # if img passed in is a file path/URL
    if type(img) is str:
        # if its a url
        if "http" in img:
            res = requests.get(img)
            # grab the binary data string from the response and convert to bytes --> RGB image  
            image = PIL.Image.open(io.BytesIO(res.content)).convert('RGB')
        else:
            image = PIL.Image.open(img)
    
    # .size is the same as .shape in PIL
    if image.size.max() > max_size:
        size = max_size
    else:
        size = image.size.max()
    
    if shape is not None:
        size = shape

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), 
                                                        (0.229, 0.224, 0.225))]) 

    image = transform(image)[:3,:,:].unsqueeze(0) 

    return image


def im_convert(tensor):

    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.255)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return image 

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    #  Need the layers for the content and style representations of an image, 
    # referencing this paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation, the rest is style
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    # name is just numbers (i.e 0,1,2,3 etc...)
    for name, layer in model._modules.items():
        x = layer(x)
        # if the layer number (aka name) is within the above defined layers, save the weights in the feature dict (i.e conv1_1: weight_tensor)
        if name in layers:
            features[layers[name]] = x
    # return feature object with all the relevant feature tensors
    return features
    

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix by matrix multiplication of the weight tensor by its transpose
    gram = torch.mm(tensor, tensor.t())
    
    return gram 