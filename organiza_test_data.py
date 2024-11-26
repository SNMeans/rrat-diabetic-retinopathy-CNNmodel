import os
import pandas as pd
import shutil

# Paths
test_dir = 'data/diabetic-retinopathy-detection/test'  # Path to your test images
labels_path = 'data/diabetic-retinopathy-detection/trainLabels.csv'  # Path to trainLabels.csv

# Load labels
print("Loading trainLabels.csv...")
labels_df = pd.read_csv(labels_path)

# Ensure subdirectories exist for each class
print("Creating subdirectories for each class...")
for class_label in labels_df['level'].unique():
    class_dir = os.path.join(test_dir, str(class_label))
    os.makedirs(class_dir, exist_ok=True)

# Move files into their respective class subdirectories
print("Organizing images into class subdirectories...")
for _, row in labels_df.iterrows():
    image_name = row['image']  # Get image filename (without extension)
    class_label = row['level']  # Get class label (0, 1, 2, etc.)
    
    src_path = os.path.join(test_dir, f"{image_name}.jpeg")  # Source path
    dst_path = os.path.join(test_dir, str(class_label), f"{image_name}.jpeg")  # Destination path

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)  # Move file to the class folder
    else:
        print(f"Image not found: {src_path}")

print("Image organization complete!")
