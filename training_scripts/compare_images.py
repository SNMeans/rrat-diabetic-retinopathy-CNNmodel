import matplotlib.pyplot as plt
import cv2
import os

# Paths
ORIGINAL_IMAGE_PATH = r"C:\Users\sumin\code\ResNet\CNN-model\data\diabetic-retinopathy-detection\train\2\15_right.jpeg"
AUGMENTED_IMAGES_DIR = r"C:\Users\sumin\code\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_augmented\2"

# Augmented image file names
AUGMENTED_IMAGES = [
    "15_right_rot90.jpeg",
    "15_right_rot180.jpeg",
    "15_right_rot270.jpeg",
    "15_right_mir.jpeg"
]

def compare_original_and_augmented(original_path, augmented_dir, augmented_files):
    """
    Compare the original image and its augmented versions side by side.
    
    Args:
        original_path (str): Path to the original image.
        augmented_dir (str): Directory containing augmented images.
        augmented_files (list): List of augmented image file names.
    """
    # Load the original image
    original_img = cv2.imread(original_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load augmented images
    augmented_imgs = []
    for img_file in augmented_files:
        img_path = os.path.join(augmented_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_imgs.append((img_file, img))
    
    # Plot original and augmented images
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(1, len(augmented_imgs) + 1, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Augmented images
    for idx, (name, img) in enumerate(augmented_imgs):
        plt.subplot(1, len(augmented_imgs) + 1, idx + 2)
        plt.imshow(img)
        plt.title(name.split('.')[0])  # Display name without extension
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
compare_original_and_augmented(ORIGINAL_IMAGE_PATH, AUGMENTED_IMAGES_DIR, AUGMENTED_IMAGES)
