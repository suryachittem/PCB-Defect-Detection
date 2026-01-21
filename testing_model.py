"""
test_efficientnet_b4.py

Inference & Evaluation script for PCB defect detection using trained EfficientNet-B4.
"""

import os
from pathlib import Path
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import timm
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Preprocessing
# -----------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# -----------------------
# Load model
# -----------------------
def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt['classes']
    num_classes = len(classes)
    
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, classes

# -----------------------
# Predict single image
# -----------------------
def predict_image(model, img_path, transform, classes, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
    
    return classes[pred_idx]

# -----------------------
# Annotate image
# -----------------------
def annotate_image(img_path, pred_class, out_dir):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Pred: {pred_class}", fill="red", font=font)
    
    fname = os.path.basename(img_path)
    save_path = os.path.join(out_dir, fname)
    img.save(save_path)

# -----------------------
# Evaluation on labeled test set
# -----------------------
def evaluate_model(model, test_dir, device):
    transform = get_transform()
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = test_dataset.classes

    # Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save CSV
    df = pd.DataFrame({
        "true_label": [class_names[i] for i in all_labels],
        "predicted_label": [class_names[i] for i in all_preds]
    })
    df.to_csv("test_eval_results.csv", index=False)
    print("\n✅ Evaluation results saved to test_eval_results.csv")

# -----------------------
# Main inference
# -----------------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print("Using device:", device)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, classes = load_model(args.checkpoint, device=device)
    transform = get_transform()
    
    if args.evaluate:
        # Evaluate on labeled test set
        evaluate_model(model, args.test_folder, device)
    else:
        # Inference on images (no labels)
        results = []
        image_files = [f for f in os.listdir(args.test_folder) if f.lower().endswith(('.jpg','.png'))]
        for fname in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(args.test_folder, fname)
            pred = predict_image(model, img_path, transform, classes, device=device)
            results.append((fname, pred))
            
            if args.save_annotated:
                annotate_image(img_path, pred, out_dir)
        
        df = pd.DataFrame(results, columns=['image', 'predicted_class'])
        df.to_csv(out_dir / 'test_predictions.csv', index=False)
        
        print(f"Inference complete. Predictions saved to {out_dir / 'test_predictions.csv'}")
        if args.save_annotated:
            print(f"Annotated images saved to {out_dir}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best_checkpoint.pth', help='Path to trained checkpoint')
    parser.add_argument('--test_folder', type=str, required=True, help='Folder containing test images')
    parser.add_argument('--output_dir', type=str, default='test_outputs', help='Where to save predictions/annotated images')
    parser.add_argument('--save_annotated', action='store_true', help='Save images with predicted labels')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU even if GPU is available')
    parser.add_argument('--evaluate', action='store_true', help='Run full evaluation on labeled test set')
    args = parser.parse_args()
    
    main(args)
