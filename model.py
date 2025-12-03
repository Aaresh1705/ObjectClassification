import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights


class VGG16Detector(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        # Load VGG16 backbone
        if pretrained:
            self.backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            self.backbone = vgg16(weights=None)

        # Freeze feature extractor if pretrained
        for param in self.backbone.features.parameters():
            param.requires_grad = not pretrained

        # Replace the classifier (fc layers)
        # We keep fc1 + fc2, but remove fc3
        in_features = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(12800, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        # ---- Two Heads ----

        # 1. Classification head
        self.classifier_head = nn.Linear(4096, num_classes)

        # 2. Bounding box regression head
        # Regression to 4 numbers: [x, y, w, ]
        self.bbox_head = nn.Linear(4096, 4)

    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        class_logits = self.classifier_head(x)
        bbox = self.bbox_head(x)

        return class_logits, bbox


def get_vgg16_model(pretrained: bool=False, num_classes: int=2) -> nn.Module:
    return VGG16Detector(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    get_vgg16_model(pretrained=True)
