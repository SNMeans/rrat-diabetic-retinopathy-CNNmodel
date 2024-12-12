import os
import shutil
import pandas as pd

# Paths
test_images_dir = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\test"
test_labels_csv = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\testLabels15_cleaned.csv"
organized_test_dir = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\organized_test"

# Create subfolders for each class
os.makedirs(organized_test_dir, exist_ok=True)
for i in range(5):  # Classes 0 to 4
    os.makedirs(os.path.join(organized_test_dir, str(i)), exist_ok=True)

# Read testLabels CSV
df = pd.read_csv(test_labels_csv)

# Move images into respective class subfolders
for _, row in df.iterrows():
    image_name, label = row['image'], str(row['level'])
    src_path = os.path.join(test_images_dir, image_name + ".jpeg")  # Assuming .jpeg extension
    dest_path = os.path.join(organized_test_dir, label, image_name + ".jpeg")

    # Move image if it exists
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"Warning: {src_path} does not exist.")

