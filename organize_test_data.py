import os
import pandas as pd
import shutil

# Use absolute paths for both the test directory and labels file
test_dir = os.path.abspath('data/diabetic-retinopathy-detection/test')
labels_path = os.path.abspath('data/diabetic-retinopathy-detection/trainLabels_updated.csv')

# Helper function to find the correct file path
def get_file_path(image_name, directory):
    """
    Check if an image file exists with the .jpeg or .jpg extension.
    """
    for ext in ['jpeg', 'jpg']:
        potential_path = os.path.join(directory, f"{image_name}.{ext}")
        if os.path.exists(potential_path):
            return potential_path
    return None

# Load trainLabels_updated.csv
if os.path.exists(labels_path):
    print("Loading trainLabels_updated.csv...")
    labels_df = pd.read_csv(labels_path)
else:
    print(f"File not found: {labels_path}")
    exit()

# Ensure subdirectories exist for each class (0, 1, 2, etc.)
print("Creating subdirectories...")
for class_label in labels_df['level'].unique():
    class_dir = os.path.join(test_dir, str(class_label))
    os.makedirs(class_dir, exist_ok=True)
    print(f"Ensured directory exists: {class_dir}")

# Move files into class subdirectories
print("Organizing images into class subdirectories...")
missing_images = []
for _, row in labels_df.iterrows():
    image_name = row['image'].split('.')[0]  # Remove extension if needed
    class_label = row['level']  # Example: 0, 1, 2, etc.

    # Find the image file with the correct extension
    src_path = get_file_path(image_name, test_dir)
    if src_path:
        # Build the destination path
        dst_path = os.path.join(test_dir, str(class_label), os.path.basename(src_path))
        # Move the file
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} -> {dst_path}")
    else:
        missing_images.append(image_name)
        print(f"Image not found: {image_name}")

# Print a summary
print("Image organization complete!")
if missing_images:
    print(f"Missing images: {len(missing_images)}")
    print(f"Examples of missing files: {missing_images[:10]}")
else:
    print("All images were successfully organized!")

