import torch
import torch.nn as nn
import torchvision.models as models

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)