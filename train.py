import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, classification_report, confusion_matrix
import seaborn as sns

import config
from models.fusion_model import FusionModel
from utils.preprocessing import get_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(preds_cls, labels_cls, preds_reg, labels_reg, preds_risk, labels_risk):
    _, preds_cls_idx = torch.max(preds_cls, 1)
    acc = accuracy_score(labels_cls.cpu(), preds_cls_idx.cpu())
    rmse = np.sqrt(mean_squared_error(labels_reg.cpu().numpy(), preds_reg.cpu().numpy()))
    preds_risk_bin = (torch.sigmoid(preds_risk) > 0.5).int()
    f1 = f1_score(labels_risk.cpu(), preds_risk_bin.cpu(), zero_division=0)
    return acc, rmse, f1, preds_cls_idx.cpu().numpy(), labels_cls.cpu().numpy()


def save_training_curves(history, save_dir):
    """Save loss and metric curves as a publication-quality plot."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Modal Fusion Model — Training Summary', fontsize=16, fontweight='bold', y=1.01)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    axes[0, 0].set_title('Combined Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Classification Accuracy
    axes[0, 1].plot(epochs, history['val_acc'], 'g-o', markersize=4, label='Val Accuracy')
    axes[0, 1].set_title('Land Use Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Regression RMSE
    axes[1, 0].plot(epochs, history['val_rmse'], 'm-o', markersize=4, label='Val RMSE')
    axes[1, 0].set_title('Vegetation Score RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Risk F1
    axes[1, 1].plot(epochs, history['val_f1'], 'c-o', markersize=4, label='Val F1')
    axes[1, 1].set_title('Environmental Risk F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Training curves saved to {curve_path}")


def save_confusion_matrix(all_labels, all_preds, save_dir):
    """Save per-class confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=config.EUROSAT_CLASSES,
        output_dict=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=config.EUROSAT_CLASSES,
        yticklabels=config.EUROSAT_CLASSES,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix — Land Use Classification', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Per-class F1 bar chart
    class_f1 = [report[cls]['f1-score'] for cls in config.EUROSAT_CLASSES]
    colors = ['#2ecc71' if f >= 0.9 else '#f39c12' if f >= 0.75 else '#e74c3c' for f in class_f1]
    bars = axes[1].bar(config.EUROSAT_CLASSES, class_f1, color=colors, edgecolor='white', linewidth=0.8)
    axes[1].set_title('Per-Class F1 Score', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Land Use Class')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim([0, 1.1])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.4, label='0.90 target')
    axes[1].legend()
    for bar, val in zip(bars, class_f1):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Confusion matrix saved to {cm_path}")

    # Also save report as JSON for README generation
    report_path = os.path.join(save_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[*] Classification report saved to {report_path}")

    return report


def train():
    set_seed(config.SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Multi-Modal Satellite Intelligence System — Training")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {config.EPOCHS}")
    print(f"  Batch   : {config.BATCH_SIZE}")
    print(f"  LR      : {config.LEARNING_RATE}")
    print(f"  Seed    : {config.SEED}")
    print(f"{'='*60}\n")

    train_loader, val_loader = get_dataloaders(
        config.RAW_DATA_DIR,
        config.BATCH_SIZE,
        config.NUM_WORKERS
    )

    model = FusionModel(num_classes=config.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}\n")

    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss()
    crit_risk = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing scheduler for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_rmse': [], 'val_f1': []
    }

    for epoch in range(config.EPOCHS):
        # ── Training ──────────────────────────────────────────────
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

            loss_cls = crit_cls(class_logits, labels_cls)
            loss_reg = crit_reg(veg_scores, labels_reg)
            loss_risk = crit_risk(risk_logits, labels_risk)
            total_loss = loss_cls + config.LAMBDA_REGRESSION * loss_reg + config.LAMBDA_RISK * loss_risk

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validation ─────────────────────────────────────────────
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

        acc, rmse, f1, epoch_preds, epoch_labels = calculate_metrics(
            torch.cat(all_cls_preds), torch.cat(all_cls_labels),
            torch.cat(all_reg_preds), torch.cat(all_reg_labels),
            torch.cat(all_risk_preds), torch.cat(all_risk_labels)
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_rmse'].append(rmse)
        history['val_f1'].append(f1)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{config.EPOCHS}]  "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | RMSE: {rmse:.4f} | F1: {f1:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.MODEL_DIR, "best_fusion_model.pth")
            torch.save(model.state_dict(), model_path)
            # Save final epoch labels/preds for confusion matrix
            best_preds = epoch_preds
            best_labels = epoch_labels
            print(f"  --> Best model saved (val_loss={val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"  Training Complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    # Save training artifacts
    save_training_curves(history, results_dir)
    report = save_confusion_matrix(best_labels, best_preds, results_dir)

    # Print final per-class summary
    print("\n  Per-Class Classification Report (Best Epoch):")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*55}")
    for cls in config.EUROSAT_CLASSES:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f = report[cls]['f1-score']
        print(f"  {cls:<25} {p:>10.3f} {r:>8.3f} {f:>8.3f}")
    print(f"\n  Overall Accuracy : {report['accuracy']:.4f}")
    print(f"  Macro F1         : {report['macro avg']['f1-score']:.4f}")

    # Save history for potential later analysis
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[*] Training history saved to {history_path}")


if __name__ == "__main__":
    train()