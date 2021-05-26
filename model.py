from config import ModelType
from torchvision import models

def loadModel(model_type):
    if model_type == ModelType.VGG16:
        model = None
    elif model_type == ModelType.ResNet152:
        model = models.resnet152(pretrained=True)
    elif model_type == ModelType.DenseNet121:
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Invalid model type!')
    return model

