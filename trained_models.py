# -*- coding: utf-8 -*-

from keras.applications import ResNet50, VGG16 

def resnet_50(pretrained_models_path, pretrained_models):
    """

    Parameters
    ----------
    pretrained_models_path : string
    pretrained_models : dictionary

    Returns
    -------
    net : object of the model. This can now be used for transfer learning.

    """
    weights_path = pretrained_models_path + pretrained_models["resnet_50"]["weights"]
    net = ResNet50(include_top = False, weights = weights_path)
    for layer in net.layers:
        layer.trainable = False
    return net

def vgg_16(pretrained_models_path, pretrained_models):
    
    """

    Parameters
    ----------
    pretrained_models_path : string
    pretrained_models : dictionary

    Returns
    -------
    net : object of the model. This can now be used for transfer learning.

    """
    weights_path = pretrained_models_path + pretrained_models["vgg_16"]["weights"]
    net = VGG16(include_top = False, weights = weights_path)
    for layer in net.layers:
        layer.trainable = False
    return net    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        