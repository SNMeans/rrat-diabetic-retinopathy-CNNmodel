# main.py
# Purpose: Explore and understand the diabetic retinopathy dataset.
# Uses functions from utils.py for preprocessing and visualization.
# Includes dataset statistics, bar charts, and image visualization.

import os
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, create_generators, plot_class_distribution, visualize_images

# Paths
train_labels_path = 'data/diabetic-retinopathy-detection/trainLabels.csv'
train_dir = 'data/diabetic-retinopathy-detection/train'

#Debugging :Confirm paths
print(f"Train labels path: {train_labels_path}")
print(f"Training directory: {train_dir}")

# Load and preprocess data
labels_df = load_and_preprocess_data(train_labels_path, train_dir)

# Print total image count and class distribution
total_images = len(labels_df)
class_distribution = labels_df['Labels'].value_counts()
print(f"Total number of images: {total_images}")
print("Distribution by class:")
print(class_distribution)

# Plot class distribution as a bar chart
plot_class_distribution(labels_df)

# Visualize sample images from each class
visualize_images(labels_df, train_dir)

# Create data generators
train_generator, validation_generator, test_generator = create_generators(labels_df)

# Display one batch of images from the training generator
batch = next(train_generator)
images, labels = batch
plt.figure(figsize=(12, 12))
for i in range(9):  # Show 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title("Class: " + str(labels[i].argmax()))
    plt.axis('off')
plt.show()
