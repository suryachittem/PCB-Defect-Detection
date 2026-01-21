"""
train_efficientnet_b4.py

Training EfficientNet-B4 on defect ROIs (128x128).
Features:
- timm EfficientNet-B4 (fallback to torchvision)
- Augmentations (train/val transforms)
- Adam optimizer, CrossEntropyLoss
- ReduceLROnPlateau scheduler
- Mixed precision training (torch.cuda.amp)
- Checkpointing, early stopping
- Accuracy, loss logging, matplotlib plots, confusion matrix (sklearn)
"""

import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import timm

# -----------------------
# Utilities & reproducibility
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# Model builder
# -----------------------
def build_model(num_classes, pretrained=True, device='cuda'):
    try:
        # timm variant (recommended)
        model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
    except Exception:
        # fallback: torchvision
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features if hasattr(model.classifier, "__getitem__") else model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    return model.to(device)

# -----------------------
# Train / Validate loops
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100.0*correct/total)

    return running_loss / total, correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Valid', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            pbar.set_postfix(loss=running_loss/total, acc=100.0*correct/total)

    return running_loss / total, correct / total, y_true, y_pred

# -----------------------
# Plot helpers
# -----------------------
def plot_metrics(history, outdir):
    df = pd.DataFrame(history)

    # Loss
    plt.figure(figsize=(8,6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'loss_curve.png'))
    plt.close()

    # Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'acc_curve.png'))
    plt.close()

# -----------------------
# Main training routine
# -----------------------
def main(args):
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print("Using device:", device)

    # Directories
    train_dir, val_dir, test_dir = Path(args.data_root)/'train', Path(args.data_root)/'val', Path(args.data_root)/'test'
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(val_dir), transform=val_transforms)
    test_dataset  = datasets.ImageFolder(str(test_dir), transform=val_transforms)

    print("Classes:", train_dataset.classes)
    num_classes = len(train_dataset.classes)

    # Dataloaders
    nw = min(args.num_workers, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device=='cuda' and args.use_amp else None

    best_val_acc, epochs_no_improve = 0.0, 0
    history = defaultdict(list)

    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, y_true_val, y_pred_val = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch} => train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'classes': train_dataset.classes
            }, outdir / 'best_checkpoint.pth')
            print("Saved best_checkpoint.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"No improvement for {epochs_no_improve} epochs. Early stopping.")
            break

    # Final plots
    plot_metrics(history, outdir)

    # Evaluate on test set
    print("\n=== Evaluating on test set ===")
    ckpt = torch.load(outdir / 'best_checkpoint.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    test_loss, test_acc, y_true_test, y_pred_test = validate_one_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    cm = confusion_matrix(y_true_test, y_pred_test)
    print("\nClassification Report:\n", classification_report(y_true_test, y_pred_test, target_names=train_dataset.classes, digits=4))

    # Save confusion matrix plot
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix'); plt.colorbar()
    tick_marks = np.arange(len(train_dataset.classes))
    plt.xticks(tick_marks, train_dataset.classes, rotation=45)
    plt.yticks(tick_marks, train_dataset.classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outdir / 'confusion_matrix.png')
    plt.close()

    # Save history CSV
    pd.DataFrame(history).to_csv(outdir / 'training_history.csv', index=False)
    print("Training complete. Artifacts saved to:", outdir)

# -----------------------
# CLI
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='root folder containing train/val/test')
    parser.add_argument('--output_dir', type=str, default='outputs', help='where to save checkpoints and plots')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=6)
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--no_pretrained', action='store_true', help='disable pretrained weights')
    parser.add_argument('--force_cpu', action='store_true', help='force cpu')
    args = parser.parse_args()
    main(args)
