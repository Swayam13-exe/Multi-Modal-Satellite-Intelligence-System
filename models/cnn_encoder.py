import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=512):
        super(CNNEncoder, self).__init__()
        # Load a pretrained ResNet18
        # weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Extract all layers except the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 natively outputs 512-dim features before the final FC
        # We can pass it through an adaptive pool to be safe, but resnet does this natively.
        # We add an optional projection if output_dim differs, but keeping it 512 is standard.
        self.output_dim = 512
        
        self.projection = nn.Identity() if output_dim == 512 else nn.Linear(512, output_dim)

    def forward(self, x):
        # x shape: (N, 3, 224, 224)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (N, 512)
        x = self.projection(x)
        return x
