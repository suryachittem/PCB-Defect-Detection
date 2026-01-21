import os, shutil
import splitfolders  # pip install split-folders

input_folder = r"C:\Users\SURYA\Desktop\PCB_DATASET\defect_rois"
output_folder = r"C:\Users\SURYA\Desktop\PCB_DATASET\defect_rois_split"

# This will create train/val/test split (80/10/10)
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1))
