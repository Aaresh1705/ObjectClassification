import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights


def get_vgg16_model(pretrained: bool=False, custom_weights: str='') -> nn.Module:
    """
    Returns the VGG16 model
    :param pretrained: Loads the pretrained model if True
    :param custom_weights: If pretrained is False and custom_weights are given then they will be loaded into the model.
    :return: The VGG16 model
    """

    def get_model(weights) -> nn.Module:
        model = vgg16(weights=weights)
        # print(model.features)
        # print(model.classifier)

        """
        160/2⁵ = 5
        5 * 5 * 512 = 12800
        """

        model.classifier[0] = nn.Linear(12800, 4096)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    if pretrained:
        model = get_model(weights=VGG16_Weights.DEFAULT)

        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = get_model(weights=None)

        for param in model.features.parameters():
            param.requires_grad = True

    if custom_weights != '':
        model.load_state_dict(torch.load(custom_weights, weights_only=True))
        print('Loaded custom weights:', custom_weights)

    return model

if __name__ == '__main__':
    get_vgg16_model(pretrained=True)
