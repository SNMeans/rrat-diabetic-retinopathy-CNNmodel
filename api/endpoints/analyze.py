import os
import sys
from io import BytesIO

project_root = r"C:\Users\sumin\code\ResNet\CNN-model"
sys.path.append(project_root)

import numpy as np
import requests
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv

#from training_scripts.single_image_test import predict_single_image

# Load environment variables 
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

#Initialize the router
router = APIRouter()

# Request schema
class ImageRequest(BaseModel):
    image_url: str
    image_id: str

# Load the model
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")
def predict_single_image(image_url, model):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from URL: {response.status_code}")
    
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence


# API Endpoint/Function
@router.post("/")
def analyze_image(image_data: ImageRequest):
    if not image_data.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")
    
    try:
        predicted_class, confidence = predict_single_image(image_data.image_url, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
    prognosis = class_labels.get(predicted_class, "Unknown")
    
    return {"image_id": image_data.image_id, "prognosis": prognosis, "confidence": round(float(confidence), 2)}

