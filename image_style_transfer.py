import torch 
from torch import optim
import PIL.Image
import utils 
import model 
import cv2 

STEPS = 2500 

model = model.getModel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

content_image = utils.load_image('https://www.rover.com/blog/wp-content/uploads/2019/01/6342530545_45ec8696c8_b-960x540.jpg').to(device)

style_image = utils.load_image('https://images2.minutemediacdn.com/image/upload/c_crop,h_1595,w_2835,x_0,y_408/f_auto,q_auto,w_1100/v1556647489/shape/mentalfloss/62280-mona_lisa-wiki.jpg').to(device)

# get content and style features only once before training
content_features = utils.get_features(content_image, model)
style_features = utils.get_features(style_image, model)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: utils.gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content_image.clone().requires_grad_(True).to(device)


# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.3,
                 'conv4_1': 0.1,
                 'conv5_1': 0.1}

content_weight = 1  # alpha
style_weight = 1e6  # beta

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)


for ii in range(1, STEPS+1):
    
    # get the features from your target image
    target_features = utils.get_features(target, model)
    
    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = utils.gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
        
    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

cv2.imwrite('saved_images/final_image.jpg', utils.im_convert(target))