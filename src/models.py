import torch
import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        n_features = self.base.fc.in_features
        self.base.fc = nn.Linear(n_features, num_classes)
    def forward(self, x) -> None:
        return self.base(x)
    
class Densenet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.densenet121(pretrained=True)
        n_features = self.base.classifier.in_features
        self.base.classifier = nn.Linear(n_features, num_classes)
    def forward(self, x) -> None:
        return self.base(x)
    
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.vgg16(pretrained=True)
        self.base.classifier[6] = nn.Linear(4096, num_classes)
    def forward(self, x) -> None:
        return self.base(x)
    
class Effnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        self.base.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    def forward(self, x) -> None:
        return self.base(x)

        