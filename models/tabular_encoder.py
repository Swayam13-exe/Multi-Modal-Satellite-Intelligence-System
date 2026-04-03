import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super(TabularEncoder, self).__init__()
        
        # 2-Layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x shape: (N, input_dim) - typically [lat_norm, lon_norm, sin_mo, cos_mo]
        Returns: (N, output_dim)
        """
        return self.network(x)
