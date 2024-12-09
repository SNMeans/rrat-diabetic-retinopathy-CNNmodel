import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from io import BytesIO

# Path to your trained model and the test image
MODEL_PATH = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\models\resnet50_diabetic_retinopathy_20241209_114444.h5"
IMAGE_URL = "https://res.cloudinary.com/rrat-dkcatdj1w/image/upload/v1733752972/rrat/retina_photos/os--775fafc5-a252-4598-8a14-f303f2e85212--2024-12-09--14:02.jpg"

# Load the trained model
model = load_model(MODEL_PATH)

def predict_single_image(image_url, model):
    """
    Predicts the class of a single image.

    Args:
    - img_path (str): Path to the image file.
    - model (tf.keras.Model): Trained model.

    Returns:
    - predicted_class (int): Predicted class index.
    - confidence (float): Confidence of the prediction.
    """
     # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from URL: {response.status_code}")
    
    # Open and preprocess the image
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence


# Predict the class of the single image
predicted_class, confidence = predict_single_image(IMAGE_URL, model)

# Define class labels
class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}

# Print the prediction
print(f"Single Image Prediction:")
print(f"Image URL: {IMAGE_URL}")
print(f"Predicted Class: {class_labels.get(predicted_class, 'Unknown')} ({predicted_class}), Confidence: {confidence:.2f}")
