# 11/24/24 Script to verify that the best model was saved

from tensorflow.keras.models import load_model

# Path to the saved model
model_path = "CNN-model/models/resnet50_best_model.h5"

# Verify loading the model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
    print(model.summary())
except Exception as e:
    print(f"Error loading model: {e}")
