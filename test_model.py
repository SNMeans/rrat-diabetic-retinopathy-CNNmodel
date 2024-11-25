# Purpose: Evaluate the trained ResNet50 model on test data and save predictions.
# Outputs: Test accuracy, loss, and predictions saved to a CSV file.

import tensorflow as tf
from utils import create_test_generator
import numpy as np
import os

# Paths to model and test data
model_path = 'CNN-model/models/resnet50_best_model.h5'
test_dir = 'CNN-model/data/diabetic-retinopathy-detection/test'

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load test data
print("Loading test data...")
test_generator = create_test_generator(test_dir)

# Evaluate the model on the test data
print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on test data
print("Running predictions...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

# Save predictions to a CSV file
output_path = os.path.abspath('test_predictions.csv')
try:
    np.savetxt(output_path, predicted_classes, delimiter=',', fmt='%d')
    print(f"Predictions saved to {output_path}")
except Exception as e:
    print(f"Error saving predictions: {e}")
