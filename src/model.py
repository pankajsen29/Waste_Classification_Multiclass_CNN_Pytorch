#########################################
#Step 2: Model initialization (ResNet18 + transfer learning)
#########################################

import torch
import torch.nn as nn
import torchvision.models as models


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#ResNet18 feature size: 512
def get_resnet18(num_classes, feature_extract=True):
    """
    Initializes a pretrained ResNet18 (with transfer learning) for multi-class classification.

    Args:
        num_classes (int): number of output classes
        feature_extract (bool): True means -> use the pretrained network as a fixed feature extractor; in other words, freeze pretrained layers
        (feature_extract answers: Do I want to reuse pretrained features as-is, or do I want to adapt them by training?)
    Returns:
        model (nn.Module)
    """
    
    '''Loads ImageNet-pretrained weights; CNN backbone ready'''
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    '''Freeze pretrained layers (Optimizer updates only required layers) (feature extraction mode),
    This ensures: backbone weights are frozen, optimizer won't touch them,
    this prevents modifying learned ImageNet features.'''
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False #don't update these weights during backpropagation

    '''Custom classifier for waste categories,
       i.e., replace the final classification layer
       why: original fc = 1000 classes (ImageNet), our task = waste categories
    '''
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    #Ensuring classifier is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_loss_function():
    """
    Loss function for multi-class classification
    """
    return nn.CrossEntropyLoss()

#lr=1e-3
def get_optimizer(model, lr=0.001):
    """
    Optimizer that updates only trainable parameters (or only the classifier head (or unfrozen layers))
    """
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.Adam(params_to_update, lr=lr)