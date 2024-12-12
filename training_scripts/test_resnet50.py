import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Paths
MODEL_PATH = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\models\resnet50_diabetic_retinopathy.h5"
TEST_DIR = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\organized_test"
SINGLE_IMAGE_PATH = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\organized_test\2\4_left.jpeg"

# Load the trained model
model = load_model(MODEL_PATH)

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),  # Switching back to 224x224
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Get true labels and predictions
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys()))

# Generate confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

# Single image prediction
def predict_single_image(img_path, model):
    """
    Predicts the class of a single image.

    Args:
    - img_path (str): Path to the image file.
    - model (tf.keras.Model): Trained model.

    Returns:
    - predicted_class (int): Predicted class index.
    - confidence (float): Confidence of the prediction.
    """
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Resize back to 224x224
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence

# Test with a single image
predicted_class, confidence = predict_single_image(SINGLE_IMAGE_PATH, model)
class_labels = {v: k for k, v in test_generator.class_indices.items()}  # Reverse the class indices
print(f"Single Image Prediction:")
print(f"Image: {SINGLE_IMAGE_PATH}")
print(f"Predicted Class: {class_labels[predicted_class]} ({predicted_class}), Confidence: {confidence:.2f}")

