import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from .feature_engineering import extract_meta_features, derive_vegetation_proxy

class MultiModalEuroSAT(Dataset):
    def __init__(self, root_dir, split='train', transform=None, download=True):
        """
        PyTorch Dataset for EuroSAT with synthetic spatial-temporal metadata.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load EuroSAT dataset
        self.eurosat = datasets.EuroSAT(root=self.root_dir, download=download)
        
        # In a real scenario, we'd split the data logically.
        # Here we randomly sample indices for train/val.
        np.random.seed(42)
        dataset_size = len(self.eurosat)
        indices = np.random.permutation(dataset_size)
        split_idx = int(0.8 * dataset_size)
        
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # Pre-generate synthetic tabular targets to keep them deterministic per object
        self.synthetic_metadata = []
        for _ in range(len(self.indices)):
            lat = random.uniform(-90.0, 90.0)
            lon = random.uniform(-180.0, 180.0)
            month = random.randint(1, 12)
            self.synthetic_metadata.append((lat, lon, month))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.eurosat[real_idx]
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        # Get synthetic metadata
        lat, lon, month = self.synthetic_metadata[idx]
        meta_features = extract_meta_features(lat, lon, month)
        
        # Derive synthetic targets:
        # 1. Vegetation Health Score (Regression): Base it on image greenness + noise
        veg_score = derive_vegetation_proxy(image_tensor)
        # Scale and add some deterministic variance
        veg_score = np.clip(veg_score * 3.0 + (lat / 180.0), 0.0, 1.0) 
        
        # 2. Risk Indicator (Binary): Some classes naturally have higher risk
        # For demonstration: Industrial, Highway, Residential trigger higher base risk.
        high_risk_classes = [3, 4, 7] # Highway, Industrial, Residential in EuroSAT
        base_risk_prob = 0.8 if label in high_risk_classes else 0.2
        risk_label = 1.0 if random.random() < base_risk_prob else 0.0
        
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)
        target = {
            'class_label': torch.tensor(label, dtype=torch.long),
            'veg_score': torch.tensor(veg_score, dtype=torch.float32),
            'risk_label': torch.tensor(risk_label, dtype=torch.float32)
        }
        
        return image_tensor, meta_tensor, target

def get_dataloaders(root_dir, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiModalEuroSAT(root_dir, split='train', transform=transform, download=True)
    val_dataset = MultiModalEuroSAT(root_dir, split='val', transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
