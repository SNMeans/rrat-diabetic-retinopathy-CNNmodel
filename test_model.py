# Purpose: Evaluate the trained ResNet50 model on test data and save predictions.
# Outputs: Test accuracy, loss, and predictions saved to a CSV file.

import tensorflow as tf
from utils import create_test_generator
import numpy as np

# Paths to model and test data
model_path = os.path.abspath('CNN-model/models/resnet50_best_model.h5')  # Absolute path to model
test_dir = os.path.abspath('data/diabetic-retinopathy-detection/test')    # Absolute path to test directory

# Debugging: Print paths
print(f"Model path: {model_path}")
print(f"Test data path: {test_dir}")

# Load the trained model
try:
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load test data
try:
    print("Loading test data...")
    test_generator = create_test_generator(test_dir)
    print(f"Test data loaded successfully! Found {test_generator.samples} samples.")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit(1)

# Evaluate model on test data
print("Evaluating model on test data...")
try:
    results = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)

# Predict on test data
print("Running predictions...")
try:
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
    print("Predictions completed!")

# Save predictions to a CSV file
output_path = os.path.abspath('test_predictions.csv')
try:
    np.savetxt(output_path, predicted_classes, delimiter=',', fmt='%d')
    print(f"Predictions saved to {output_path}")
except Exception as e:
    print(f"Error saving predictions: {e}")
