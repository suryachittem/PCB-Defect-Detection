import cv2
import os
import subprocess

# ----------- PATH CONFIGURATION -----------
# Input difference images (from Module 1 output)
DIFF_IMAGES_ROOT = r"C:\Users\SURYA\Desktop\PCB_DATASET\output_diff"

# Output folders
ROI_OUTPUT_ROOT = r"C:\Users\SURYA\Desktop\PCB_DATASET\defect_rois"
VIS_OUTPUT_ROOT = r"C:\Users\SURYA\Desktop\PCB_DATASET\defect_visuals"

# Create output directories if not exist
os.makedirs(ROI_OUTPUT_ROOT, exist_ok=True)
os.makedirs(VIS_OUTPUT_ROOT, exist_ok=True)


def process_image(img_path, save_folder_roi, save_folder_vis):
    """
    Process one difference image:
    - Extracts ROIs
    - Saves visualization with contours & bounding boxes
    """
    diff = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if diff is None:
        print(f"⚠️ Could not read {img_path}")
        return

    # Convert to color for drawing
    vis_img = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    # ---- Preprocessing ----
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # ---- Contour Detection ----
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore very small regions (noise)
        if w < 10 or h < 10:
            continue

        # ---- ROI Extraction ----
        roi = diff[y:y + h, x:x + w]
        roi_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_roi{roi_count}.png"
        cv2.imwrite(os.path.join(save_folder_roi, roi_filename), roi)

        # ---- Visualization ----
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 1)            # Green contour
        cv2.putText(vis_img, f"ROI {roi_count}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        roi_count += 1

    # Save visualization
    vis_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_vis.png"
    cv2.imwrite(os.path.join(save_folder_vis, vis_filename), vis_img)

    print(f"✅ {os.path.basename(img_path)}: {roi_count} ROIs extracted & visualized")


def main():
    """ Main pipeline for Module 2 (processes every image in all categories) """

    has_subfolders = any(os.path.isdir(os.path.join(DIFF_IMAGES_ROOT, d)) 
                         for d in os.listdir(DIFF_IMAGES_ROOT))

    if has_subfolders:
        # Case 1: Images are inside subfolders (per defect type)
        for defect_type in os.listdir(DIFF_IMAGES_ROOT):
            defect_folder = os.path.join(DIFF_IMAGES_ROOT, defect_type)
            if not os.path.isdir(defect_folder):
                continue

            print(f"\n🔍 Processing category: {defect_type}")

            # Create output folders for this category
            save_folder_roi = os.path.join(ROI_OUTPUT_ROOT, defect_type)
            save_folder_vis = os.path.join(VIS_OUTPUT_ROOT, defect_type)
            os.makedirs(save_folder_roi, exist_ok=True)
            os.makedirs(save_folder_vis, exist_ok=True)

            # Process each image inside the defect category
            for img_file in os.listdir(defect_folder):
                img_path = os.path.join(defect_folder, img_file)
                print(f"📂 Processing image: {img_path}")
                process_image(img_path, save_folder_roi, save_folder_vis)
    else:
        # Case 2: Images are directly inside DIFF_IMAGES_ROOT
        print(f"\n🔍 Processing images directly in {DIFF_IMAGES_ROOT}")
        for img_file in os.listdir(DIFF_IMAGES_ROOT):
            img_path = os.path.join(DIFF_IMAGES_ROOT, img_file)
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            print(f"📂 Processing image: {img_path}")
            process_image(img_path, ROI_OUTPUT_ROOT, VIS_OUTPUT_ROOT)

    print("\n🎯 Module 2 Complete: ROIs + Visualizations saved.")

    # Auto-open the visualization folder
    subprocess.Popen(f'explorer "{VIS_OUTPUT_ROOT}"')


if __name__ == "__main__":
    main()
