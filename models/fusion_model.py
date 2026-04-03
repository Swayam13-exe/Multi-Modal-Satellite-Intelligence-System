import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
from .tabular_encoder import TabularEncoder

class FusionModel(nn.Module):
    def __init__(self, num_classes=10, cnn_out=512, tab_out=128):
        super(FusionModel, self).__init__()
        
        self.cnn_encoder = CNNEncoder(pretrained=True, output_dim=cnn_out)
        self.tabular_encoder = TabularEncoder(input_dim=4, hidden_dim=64, output_dim=tab_out)
        
        fusion_dim = cnn_out + tab_out
        
        # Fusion Layer: Dense -> Dropout -> Dense
        self.fusion_block = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Output Heads
        # 1. Land Use Classification (10 classes) -> Outputs logits for CrossEntropyLoss
        self.classifier_head = nn.Linear(128, num_classes)
        
        # 2. Vegetation Health Score (Regression 0 to 1) -> Sigmoid activation
        self.regressor_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 3. Environmental Risk Indicator (Binary) -> Outputs logit for BCEWithLogitsLoss
        self.risk_head = nn.Linear(128, 1)

    def forward(self, images, metadata):
        # Extract features
        img_features = self.cnn_encoder(images)
        meta_features = self.tabular_encoder(metadata)
        
        # Concatenate features
        fused = torch.cat((img_features, meta_features), dim=1)
        
        # Pass through fusion block
        shared_rep = self.fusion_block(fused)
        
        # Branch to heads
        class_logits = self.classifier_head(shared_rep)
        veg_score = self.regressor_head(shared_rep).squeeze(1) # [N, 1] -> [N]
        risk_logits = self.risk_head(shared_rep).squeeze(1)    # [N, 1] -> [N]
        
        return class_logits, veg_score, risk_logits
