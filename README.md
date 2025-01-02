# EyeQ: Retinopathy Risk Assessment Model (RRAT Model)

This repository contains the Convolutional Neural Network (CNN) model code for **EyeQ**. This model processes retinal fundus images to predict diabetic retinopathy severity, aiding medical professionals in early detection and treatment planning.

---

## **Overview**
The CNN Model is a component of the django based EyeQ web application. It performs the following:
- **Processes**: Prepares input images with advanced preprocessing techniques.
- **Predicts**: Identifies diabetic retinopathy severity levels.
- **Integrates**: Sends predictions to the EyeQ frontend via a REST API for real-time display.
  
### **Application Interface**
Below are screenshots of the EyeQ application showcasing the homepage and login screen:
<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  <img src="https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/homepage.png" 
       alt="Homepage" style="height: 300px; width: auto; border: 2px solid #ddd; border-radius: 8px;">
  <img src="https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/login.png" 
       alt="Login Screen" style="height: 300px; width: auto; border: 2px solid #ddd; border-radius: 8px;">
</div>

1. **Left Image**: The homepage of the EyeQ application, providing access to patient uploads and results.
2. **Right Image**: The login screen, ensuring secure access for authorized medical professionals.

---

## **Repository Integration**
This repository is a submodule of the main **EyeQ** project:
- **Main Repo**: [EyeQ on GitHub](https://github.com/NeffCodes/retinopathy-risk-assessment-tool)
- **GitHub Actions**: Configured for CI/CD workflows, including automated testing and deployment.

---

## **Features**
### 1. **Image Preprocessing**
- Resizing, cropping, rotation, and augmentation using OpenCV and custom scripts.
- Normalization to standardize image data for consistent input to the model.

### 2. **Risk Prediction**
- **Model**: Fine-tuned ResNet-50 for feature extraction and classification.
- **Outputs Five risk levels**: (0) No DR, (1) Mild, (2) Moderate, (3) Severe, (4) Proliferative.


### 3. **API Integration**
- Real-time prediction API using FastAPI or Django REST Framework.
- RESTful endpoints for submitting images and retrieving predictions.

---

## **Architecture**
- **Model Training**: Developed with TensorFlow and Keras.
- **Augmentation**:
  - Applied rotation, mirroring, and cropping scripts to address dataset imbalance.
  - Oversampled smaller classes (1-4) to stabilize training and reduce bias.
- **REST API**:
  - Accepts uploaded images, processes predictions, and returns results which are displayed on the patient dashboard.

### **Below are examples of the retina photo processing workflow in the EyeQ application:**

<div style="display: flex; align-items: flex-start; gap: 20px;">
  <img src="https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/preprocess.png" alt="Uploaded Retina Photo" style="width: 300px; height: 300px;">
  <img src="https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/display.png" alt="Processed Retina Photo" style="width: 300px; height: 300px;">
</div>

1. **Left Image**: Retina photo uploaded to the application, ready for analysis.
2. **Right Image**: The processed retina photo with a result of "Moderate" severity, displayed after the prediction.



---

## **Dataset and Preprocessing**
Our training dataset originates from the 2015 Kaggle competition and contains **35,126 images** across **5 classes** (0–4). Preprocessing was essential to address challenges like:
- **Class Imbalance**: Class `0` (No DR) made up **73%** of the data (25,810 images).
- **Inconsistent Quality**: Images had varying resolutions and quality.

### **Preprocessing Steps**
1. Cropped images to **1800x1800**.
2. Resized images to **224x224**.
3. Removed black or empty images.
4. Augmented images:
   - Rotated DR images by **90°, 180°, 270°**.
   - Mirrored images to improve variability.
5. Converted all images to NumPy arrays.

| **Class** | **Original Count** | **After Augmentation** |
|-----------|--------------------|-------------------------|
| 0 (No DR) | 25,810            | 25,810 (unchanged)     |
| 1–4 (DR)  | 9,316             | 27,948                 |

---

## **Training Process**
The data was split **80/20** into training and validation sets. Key details:
- **Framework**: TensorFlow and Keras.
- **Architecture**: ResNet-50 fine-tuned for retinal image classification.
- **Optimization**:
  - Learning Rate: Tuned for stability.
  - Batch Size: Adjusted based on GPU memory.
- **Hardware**: CUDA and cuDNN utilized for GPU acceleration.

| **Metric**  | **Score** |
|-------------|-----------|
| Accuracy    | 73%       |
| Precision   | XX%       |
| Recall      | XX%       |
| F1 Score    | XX%       |

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or higher
- TensorFlow, Keras, OpenCV, NumPy, Pandas
- Kaggle API for dataset

### **Installation Steps**
1. Clone this repository:
    ```bash
    git clone https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel.git
    cd rrat-diabetic-retinopathy-CNNmodel
    ```
2. Set up a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # MacOS/Linux
    env\Scripts\activate  # Windows
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Copy and configure `.env-template`:
    ```bash
    cp .env-template .env
    ```

---

## **Usage**

1. **Download Dataset**: Use the Kaggle API to download the [Diabetic Retinopathy Detection dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).
2. **Preprocess Images**:
    ```bash
    python training_scripts/crop_resize.py
    python training_scripts/augment_images.py
    ```
3. **Train the Model**:
    ```bash
    python training_scripts/train_resnet50.py
    ```
4. **Start API Server**:
    ```bash
    python api/runserver.py
    ```

---

## **Visualization**

### **Before and After Augmentation**
This photo displays the augmentation process applied to the retina photos, showcasing preprocessing techniques like rotation and mirroring to increase variability in the dataset:
![Augmentation Fundus](https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/Rotimages.png)

### **Data Distribution**
**Before Augmentation**:
This graph displays the original data distribution, highlighting the severe imbalance in the dataset, which was heavily biased toward class 0 (no diabetic retinopathy)
![Before Augmentation](https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/BEFOREAUG.png)

**After Augmentation**:
This graph shows the data distribution after oversampling the more severe classes, ensuring a more balanced dataset for training and improving the model's performance.
![After Augmentation](https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel/blob/main/images/DRafterAug.png)

---

## **Contributing**

Please follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
    ```bash
    git commit -m 'Add feature'
    ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or inquiries, please contact us at Sumi Means or James Neff.

