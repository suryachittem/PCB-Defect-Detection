import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import timm

# ==================================
# 1. SETUP & CONFIGURATION
# ==================================
app = Flask(__name__)
CORS(app)

# --- Paths ---
MODEL_PATH = os.path.join("outputs", "best_checkpoint.pth")
# NEW: Directory to store your golden "reference" PCB images
REFERENCE_DIR = r"C:\Users\SURYA\Desktop\PCB_DATASET\PCB_USED" 

os.makedirs(REFERENCE_DIR, exist_ok=True)

# ==================================
# 2. LOAD THE TRAINED MODEL & REFERENCE IMAGES
# ==================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = []
reference_db = {} # NEW: Dictionary to hold reference image data and features

# --- Load Classification Model ---
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Classification model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load the classification model: {e}")

# --- NEW: Load Reference Images and Pre-compute Features ---
try:
    print("\nLoading reference images...")
    orb = cv2.ORB_create(nfeatures=2000) # Initialize feature detector
    if not os.listdir(REFERENCE_DIR):
        print("⚠️ WARNING: The 'reference_pcbs' directory is empty. The auto-matching will not work.")
    
    for filename in os.listdir(REFERENCE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            model_name = os.path.splitext(filename)[0]
            image_path = os.path.join(REFERENCE_DIR, filename)
            
            # Load image in both PIL and OpenCV formats
            ref_pil = Image.open(image_path).convert("RGB")
            ref_cv = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2GRAY)
            
            # Compute and store features
            kp, des = orb.detectAndCompute(ref_cv, None)
            
            reference_db[model_name] = {
                'pil_image': ref_pil,
                'descriptors': des
            }
            print(f"  -> ✅ Loaded reference: {model_name}")
except Exception as e:
    print(f"❌ ERROR: Could not load reference images: {e}")

# --- Image transformation for the model ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. FULL INFERENCE PIPELINE FUNCTIONS
# ==========================================

# NEW: Function to find the best matching reference image
def find_best_reference_image(test_img_cv, db):
    """Finds the best reference image match using ORB feature comparison."""
    if not db:
        return None # No reference images are loaded

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Convert test image to grayscale and compute features
    test_gray = cv2.cvtColor(test_img_cv, cv2.COLOR_RGB2GRAY)
    kp_test, des_test = orb.detectAndCompute(test_gray, None)

    if des_test is None:
        return None # No features found in test image

    max_matches = 0
    best_match_name = None

    for model_name, data in db.items():
        des_ref = data['descriptors']
        if des_ref is None:
            continue
        
        matches = bf.match(des_test, des_ref)
        # More matches = better similarity
        if len(matches) > max_matches:
            max_matches = len(matches)
            best_match_name = model_name
    
    print(f"🤖 Best match found: {best_match_name} with {max_matches} feature matches.")
    return db[best_match_name]['pil_image'] if best_match_name else None

# (No changes to the functions below)
def subtract_images(defect_img, ref_img):
    defect_color = np.array(defect_img.convert('RGB'))
    defect_gray = cv2.cvtColor(defect_color, cv2.COLOR_RGB2GRAY)
    ref_gray = np.array(ref_img.convert('L'))
    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))
    defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    subtracted = cv2.absdiff(defect_blur, ref_blur)
    binary = cv2.adaptiveThreshold(subtracted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)
    return defect_color, cleaned

def extract_rois(orig_img, mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        roi = orig_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        rois.append({'image': roi, 'bbox': (x, y, x+w, y+h)})
    return rois

def predict_roi(roi_img, model, class_names, transform, device):
    roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(roi_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    label = class_names[probs.argmax()]
    confidence = float(probs.max())
    return label, confidence

# ==================================
# 4. FLASK API ENDPOINTS
# ==================================

@app.route('/')
def home():
    return render_template('index.html')

# MODIFIED: The main API endpoint for the full detection pipeline.
@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    if not model:
        return jsonify({"error": "Model is not loaded on the server."}), 500
    if not reference_db:
        return jsonify({"error": "Reference images are not loaded on the server."}), 500

    # MODIFIED: Now only expects 'test_image'
    if 'test_image' not in request.files:
        return jsonify({"error": "Missing test image."}), 400

    test_file = request.files['test_image']
    
    try:
        test_pil = Image.open(test_file.stream).convert("RGB")

        # --- NEW: Auto-find the reference image ---
        test_cv = cv2.cvtColor(np.array(test_pil), cv2.COLOR_RGB2BGR)
        ref_pil = find_best_reference_image(test_cv, reference_db)
        
        if ref_pil is None:
            return jsonify({"error": "Could not find a matching reference PCB for the uploaded image."}), 400
        # --- End of new section ---

        # 1. Image Subtraction
        original_image, mask = subtract_images(test_pil, ref_pil)
        
        # 2. ROI Extraction
        rois = extract_rois(original_image, mask)
        
        if not rois:
            # Handle case where no defects are found
            _, buffer = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'annotated_image': f'data:image/png;base64,{img_base64}',
                'predictions': [{'label': 'No Defects Found', 'confidence': '100.00%'}]
            })
            
        annotated_image = original_image.copy()
        predictions = []

        # 3. Predict each ROI
        for roi_data in rois:
            label, conf = predict_roi(roi_data['image'], model, classes, transform, device)
            predictions.append({'label': label, 'confidence': f"{conf:.2%}"})
            
            x1, y1, x2, y2 = roi_data['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_image, f"{label} ({conf:.1%})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert annotated image to base64 to send in JSON
        _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'annotated_image': f'data:image/png;base64,{img_base64}',
            'predictions': predictions
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

# ==================================
# 5. RUN THE APPLICATION
# ==================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')