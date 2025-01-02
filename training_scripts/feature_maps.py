import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd

# Paths
MODEL_PATH = r"C:\Users\sumin\code\ResNet\CNN-model\models\resnet50_diabetic_retinopathy_20241209_114444.h5"
IMAGE_DIR = r"C:\Users\sumin\code\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_augmented"
LABELS_CSV = r"C:\Users\sumin\code\ResNet\CNN-model\data\diabetic-retinopathy-detection\trainLabels_balanced.csv"

# Load Model
model = load_model(MODEL_PATH)

# Extract penultimate layer for features
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)  # Penultimate layer

# Function to load images and labels
def load_images_and_labels(image_dir, target_size=(224, 224), num_images_per_class=50):
    images = []
    labels = []
    class_folders = sorted(os.listdir(image_dir))  # Ensure class order is consistent
    
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(image_dir, class_folder)
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpeg')]
        
        # Limit to num_images_per_class for t-SNE efficiency
        for img_path in image_files[:num_images_per_class]:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size) / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(class_idx)  # Assign label based on folder name
            
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images_and_labels(IMAGE_DIR, num_images_per_class=100)  # Limit images for speed

# Extract features
features = feature_extractor.predict(images)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, verbose=1)
features_2d = tsne.fit_transform(features)

# Plot t-SNE Clustering
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label="Classes")
plt.title("t-SNE Visualization of ResNet-50 Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

