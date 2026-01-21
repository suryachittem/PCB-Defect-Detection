import cv2
import os
import shutil

# good images path
good_images_path = r"C:\Users\SURYA\Desktop\PCB_DATASET\pcb_used\01.JPG"

# Path to defect images
defects_root = r"C:\Users\SURYA\Desktop\PCB_DATASET\images"

# Path to save results (always fresh new folder)
output_root = r"C:\Users\SURYA\Desktop\PCB_DATASET\output_diff"

# Remove old results if folder exists
if os.path.exists(output_root):
    shutil.rmtree(output_root)

# Create fresh output folder
os.makedirs(output_root, exist_ok=True)

# Load golden image in grayscale
golden = cv2.imread(good_images_path, cv2.IMREAD_GRAYSCALE)
if golden is None:
    raise FileNotFoundError(f"Golden image not found at {good_images_path}")

# Loop through defect categories
for defect_type in os.listdir(defects_root):
    defect_folder = os.path.join(defects_root, defect_type)
    if not os.path.isdir(defect_folder):
        continue

    print(f"Processing category: {defect_type}")

    # Create output folder for each defect type
    save_folder = os.path.join(output_root, defect_type)
    os.makedirs(save_folder, exist_ok=True)

    # Loop through images in the defect folder
    for img_file in os.listdir(defect_folder):
        img_path = os.path.join(defect_folder, img_file)
        defect = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if defect is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        # Resize defect image to match golden (if needed)
        defect = cv2.resize(defect, (golden.shape[1], golden.shape[0]))

        # Image subtraction (absolute difference)
        diff = cv2.absdiff(golden, defect)

        # Save only the difference image
        save_path = os.path.join(save_folder, f"{img_file.split('.')[0]}_diff.png")
        cv2.imwrite(save_path, diff)

print("✅ All difference images saved in:", output_root)
