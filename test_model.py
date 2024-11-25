import tensorflow as tf
from utils import create_test_generator
import numpy as np

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

# Predict on test data
print("Running predictions...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

# Save predictions
np.savetxt('test_predictions.csv', predicted_classes, delimiter=',', fmt='%d')
print("Predictions saved to test_predictions.csv")
