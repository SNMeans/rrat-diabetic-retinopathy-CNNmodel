import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
data_dir = "../data/diabetic-retinopathy-detection/train"
validation_dir = "../data/diabetic-retinopathy-detection/validation"

# Create validation directory if it doesn't exist
os.makedirs(validation_dir, exist_ok=True)

# Split data for each class
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        # Create corresponding class directory in validation
        val_class_dir = os.path.join(validation_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)

        # Split images
        images = os.listdir(class_dir)
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        # Move validation images
        for val_image in val_images:
            src_path = os.path.join(class_dir, val_image)
            dest_path = os.path.join(val_class_dir, val_image)
            shutil.move(src_path, dest_path)

        print(f"Class {class_name}: {len(val_images)} images moved to validation.")

