import os
import pandas as pd
from PIL import Image
from shutil import copyfile

# Paths
base_dir = r"C:\Users\dhana\OneDrive\Desktop\Final year project\dataset"
output_dir = r"C:\Users\dhana\OneDrive\Desktop\Final year project\yolo_dataset"

for split in ["Train", "Test"]:  # You can add "Meta" if needed
    csv_path = os.path.join(base_dir, f"{split}.csv")
    df = pd.read_csv(csv_path)
    
    for i, row in df.iterrows():
        img_path = os.path.join(base_dir, row["Path"])
        label_path = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        
        # Create output folders
        img_out_dir = os.path.join(output_dir, "images", split.lower())
        label_out_dir = os.path.join(output_dir, "labels", split.lower())
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        # Copy image
        img_out_path = os.path.join(img_out_dir, os.path.basename(img_path))
        copyfile(img_path, img_out_path)
        
        # Get image size
        width, height = row["Width"], row["Height"]
        
        # Get normalized bounding box
        x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        class_id = int(row["ClassId"])

        # Save label
        label_out_path = os.path.join(label_out_dir, label_path)
        with open(label_out_path, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print("✅ Conversion complete!")
