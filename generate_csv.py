import os
import pandas as pd

roi_root = r"C:\Users\SURYA\Desktop\PCB_DATASET\defect_rois"
csv_path = r"C:\Users\SURYA\Desktop\PCB_DATASET\labels1.csv"

rows = []
for label in os.listdir(roi_root):
    class_dir = os.path.join(roi_root, label)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append([os.path.join(label, fname), label])

df = pd.DataFrame(rows, columns=["filename", "label"])
df.to_csv(csv_path, index=False)
print(f"✅ CSV file created: {csv_path}, total samples = {len(df)}")
