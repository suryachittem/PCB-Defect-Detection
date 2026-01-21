# ===========================
# SETUP
# ===========================
#!pip install torch torchvision matplotlib pandas --quiet

import os
import pandas as pd
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import files
import zipfile

# ===========================
# CREATE FOLDERS
# ===========================
os.makedirs("PCB_DATASET/ROIs", exist_ok=True)
os.makedirs("PCB_DATASET/output", exist_ok=True)

# ===========================
# UPLOAD CSV FILE
# ===========================
print("Upload your label CSV file")
uploaded_csv = files.upload()
for fname in uploaded_csv.keys():
    os.rename(fname, "PCB_DATASET/label.csv")

CSV_PATH = "PCB_DATASET/label.csv"
OUT_DIR = "PCB_DATASET/output"

# ===========================
# UPLOAD ZIP OF ROI IMAGES
# ===========================
print("Upload ZIP file of all ROI images")
uploaded_zip = files.upload()
for fname in uploaded_zip.keys():
    zip_path = fname

# Unzip into ROIs folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("PCB_DATASET/ROIs")

# Fix extra nested ROIs folder automatically
nested_folder = os.path.join("PCB_DATASET/ROIs", "ROIs")
if os.path.exists(nested_folder):
    for f in os.listdir(nested_folder):
        os.rename(os.path.join(nested_folder, f), os.path.join("PCB_DATASET/ROIs", f))
    os.rmdir(nested_folder)

ROIS_ROOT = "PCB_DATASET/ROIs"

# ===========================
# FIX CSV PATHS AUTOMATICALLY
# ===========================
df = pd.read_csv(CSV_PATH)
df['image_name'] = df['image_name'].str.replace("\\", "/", regex=False).str.strip()

# Map lowercase filenames to actual paths
file_map = {}
for dirpath, dirnames, filenames in os.walk(ROIS_ROOT):
    for f in filenames:
        file_map[f.lower()] = os.path.join(dirpath, f)

def map_path(fname):
    fname_clean = fname.replace("\\","/").strip()
    return file_map.get(os.path.basename(fname_clean).lower(), None)

df['image_path'] = df['image_name'].apply(map_path)
df = df[df['image_path'].notnull()].reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)

print("Sample image paths after fixing:")
print(df[['image_name','image_path']].head(5))

# ===========================
# DATASET CLASS
# ===========================
class PCBRoiDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_name = row['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[label_name]
        return image, label

# ===========================
# TRAINING FUNCTION
# ===========================
def train_model(CSV_PATH, OUT_DIR, num_epochs=50, batch_size=8):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_csv = os.path.join(OUT_DIR, "train_labels.csv")
    val_csv = os.path.join(OUT_DIR, "val_labels.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print("Train size:", len(train_df), "Val size:", len(val_df))

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_dataset = PCBRoiDataset(train_csv, transform=train_transform)
    val_dataset = PCBRoiDataset(val_csv, transform=val_transform)

    # Weighted sampler
    train_labels = [train_dataset.class_to_idx[row['label']] for _, row in train_dataset.df.iterrows()]
    counts = Counter(train_labels)
    class_weights = {cls: 1.0/counts[cls] for cls in counts}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes = len(train_dataset.classes)
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else None)

    best_val_acc = 0.0
    patience, patience_counter = 5, 0
    save_path = os.path.join(OUT_DIR, "efficientnet_b4_best.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(imgs)
                _, preds = outputs.max(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.3f} val_acc {val_acc:.3f} time {(time.time()-t0):.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'classes': train_dataset.classes
            }, save_path)
            print(" ✅ Saved new best:", save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered!")
                break

    print("Training finished. Best val acc:", best_val_acc)
    return model, save_path, val_transform

# ===========================
# TEST SINGLE IMAGE
# ===========================
def test_roi_image(roi_path, model, class_names, transform, device):
    img = Image.open(roi_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]
    label, conf = top3[0]
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {label} ({conf:.2f})")
    plt.show()
    return label, conf, top3

# ===========================
# RUN TRAINING
# ===========================
model, save_path, val_transform = train_model(CSV_PATH, OUT_DIR, num_epochs=50, batch_size=8)

# ===========================
# TEST A SINGLE IMAGE
# ===========================
print("Upload a single ROI image to test")
uploaded_test = files.upload()
for fname in uploaded_test.keys():
    roi_path = fname

checkpoint = torch.load(save_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
classes = checkpoint['classes']
model.load_state_dict(checkpoint['model_state_dict'])
label, conf, top3 = test_roi_image(roi_path, model, classes, val_transform, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("✅ Final Prediction:", label, conf)
print("Top-3 Predictions:", top3)