import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

import Multi-Modal-Satellite-Intelligence-System.config
from models.fusion_model import FusionModel
from utils.preprocessing import get_dataloaders

def calculate_metrics(preds_cls, labels_cls, preds_reg, labels_reg, preds_risk, labels_risk):
    # Classification Accuracy
    _, preds_cls_idx = torch.max(preds_cls, 1)
    acc = accuracy_score(labels_cls.cpu(), preds_cls_idx.cpu())
    
    # Regression RMSE
    rmse = np.sqrt(mean_squared_error(labels_reg.cpu(), preds_reg.cpu()))
    
    # Binary Classification F1
    preds_risk_bin = (torch.sigmoid(preds_risk) > 0.5).int()
    f1 = f1_score(labels_risk.cpu(), preds_risk_bin.cpu())
    
    return acc, rmse, f1

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    train_loader, val_loader = get_dataloaders(
        config.RAW_DATA_DIR, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS
    )
    
    # Init Model
    model = FusionModel(num_classes=config.NUM_CLASSES).to(device)
    
    # Loss functions
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss()
    crit_risk = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, metadata, targets in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels_cls = targets['class_label'].to(device)
            labels_reg = targets['veg_score'].to(device)
            labels_risk = targets['risk_label'].to(device)
            
            optimizer.zero_grad()
            
            class_logits, veg_scores, risk_logits = model(images, metadata)
            
            # Compute partial losses
            loss_cls = crit_cls(class_logits, labels_cls)
            loss_reg = crit_reg(veg_scores, labels_reg)
            loss_risk = crit_risk(risk_logits, labels_risk)
            
            # Total Loss
            total_loss = loss_cls + config.LAMBDA_REGRESSION * loss_reg + config.LAMBDA_RISK * loss_risk
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_cls_preds, all_cls_labels = [], []
        all_reg_preds, all_reg_labels = [], []
        all_risk_preds, all_risk_labels = [], []
        
        with torch.no_grad():
            for images, metadata, targets in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels_cls = targets['class_label'].to(device)
                labels_reg = targets['veg_score'].to(device)
                labels_risk = targets['risk_label'].to(device)
                
                class_logits, veg_scores, risk_logits = model(images, metadata)
                
                loss_cls = crit_cls(class_logits, labels_cls)
                loss_reg = crit_reg(veg_scores, labels_reg)
                loss_risk = crit_risk(risk_logits, labels_risk)
                
                batch_loss = loss_cls + config.LAMBDA_REGRESSION * loss_reg + config.LAMBDA_RISK * loss_risk
                val_loss += batch_loss.item()
                
                all_cls_preds.append(class_logits)
                all_cls_labels.append(labels_cls)
                all_reg_preds.append(veg_scores)
                all_reg_labels.append(labels_reg)
                all_risk_preds.append(risk_logits)
                all_risk_labels.append(labels_risk)
                
        val_loss /= len(val_loader)
        
        # Calculate Validation Metrics
        acc, rmse, f1 = calculate_metrics(
            torch.cat(all_cls_preds), torch.cat(all_cls_labels),
            torch.cat(all_reg_preds), torch.cat(all_reg_labels),
            torch.cat(all_risk_preds), torch.cat(all_risk_labels)
        )
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {acc:.4f} | Val RMSE: {rmse:.4f} | Val F1: {f1:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.MODEL_DIR, "best_fusion_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[*] Best model saved at {model_path}")

if __name__ == "__main__":
    train()
