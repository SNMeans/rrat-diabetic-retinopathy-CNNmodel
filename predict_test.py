import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
model_path = r'C:\Users\sumin\Documents\VScode\ResNet\CNN-model\CNN-model\models\resnet50_best_model.h5'
test_dir = r'C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\test'
output_csv = r'C:\Users\sumin\Documents\VScode\ResNet\CNN-model\predictions.csv'

# Load the trained model
print(f"Loading the trained model from {model_path}...")
model = load_model(model_path)

# Image preprocessing settings
IMG_SIZE = 256  # Adjust this based on your model's input size
BATCH_SIZE = 32  # Optional: Use batching for large datasets

# Helper function to preprocess images
def preprocess_batch(image_paths):
    batch = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        batch.append(img_array)
    return np.array(batch)

# Get a list of all test images
print(f"Selecting 3,500 random test images from {test_dir}...")
all_image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]

# Select a random subset of 3,500 images
selected_image_files = random.sample(all_image_files, 3500)
print(f"Selected {len(selected_image_files)} images for prediction.")

# Prepare for predictions
print(f"Predicting on test images in {test_dir}...")
predictions = []
num_batches = len(selected_image_files) // BATCH_SIZE + (len(selected_image_files) % BATCH_SIZE > 0)

for batch_idx in range(num_batches):
    # Get the current batch of image paths
    batch_files = selected_image_files[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
    batch_paths = [os.path.join(test_dir, img_file) for img_file in batch_files]

    # Preprocess the batch
    batch_images = preprocess_batch(batch_paths)

    # Predict on the batch
    batch_preds = model.predict(batch_images)

    # Store predictions
    for img_file, pred in zip(batch_files, batch_preds):
        predicted_class = np.argmax(pred, axis=-1)  # Assuming a classification model
        predictions.append({'image': img_file, 'predicted_class': int(predicted_class)})

    print(f"Processed batch {batch_idx + 1}/{num_batches}...")

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

