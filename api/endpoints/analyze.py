from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import os
from dotenv import load_dotenv
from single_image_test import predict_single_image

# Load environment variables 
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

#Initialize the router
router = APIRouter()

# Request schema
class ImageRequest(BaseModel):
    image_url: str
    image_id: str

# Load the model once. Do not need to redefine predict_single_image since it is being imported
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

# Request schema
class ImageRequest(BaseModel):
    image_url: str
    image_id: str

# API Endpoint/Function
@router.post("/")
def analyze_image(image_data: ImageRequest):
    if not image_data.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")
    
    # Predict the image
    try:
        predicted_class, confidence = predict_single_image(image_data.image_url, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map prediction to class labels
    class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
    prognosis = class_labels.get(predicted_class, "Unknown")
    
    return {"image_id": image_data.image_id, "prognosis": prognosis, "confidence": round(confidence, 2)}

