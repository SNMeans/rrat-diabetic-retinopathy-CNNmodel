# Retinopathy Risk Assessment Model (RRAT Model)
This repository contains the Convolutional Neural Network (CNN) model code for the Retinopathy Risk Assessment Tool (RRAT). This model is designed to analyze retinal fundus photos to assess the risk of diabetic retinopathy in patients, assisting medical professionals with early detection.

## Overview
The Retinopathy Risk Assessment CNN Model is a component of the RRAT web application. It processes patient retinal fundus photos, identifies risk levels, and returns predictions to the RRAT frontend for diagnostic display. The model utilizes supervised learning, trained on labeled fundus photos to predict levels of diabetic retinopathy.

## Repository Integration
--submodule/Github actions?-- https://github.com/NeffCodes/retinopathy-risk-assessment-tool

## Features
- **Image Preprocessing**: Prepares patient fundus photos for input into the model.
- **Risk Prediction**: Uses a CNN to assess and predict diabetic retinopathy severity.
- **REST API**: Allows the main RRAT app to send images to the model and receive prediction results.

## Planned Architecture
- **Model Training**: Built with TensorFlow and Keras.
- **Preprocessing**: Uses OpenCV for image resizing, normalization, and other preprocessing steps.
- **Prediction API**: Offers an endpoint to connect to RRAT, handling image submissions and returning predictions.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow, Keras, OpenCV, NumPy, Pandas, and other ML libraries

### Installation Steps
1. Clone this repository:
    ```bash
    git clone https://github.com/SNMeans/rrat-diabetic-retinopathy-CNNmodel.git
    cd rrat-diabetic-retinopathy-CNNmodel
    ```
2. Set up a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # For MacOS/Linux
    env\Scripts\activate  # For Windows
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Download Dataset**: Use the Kaggle API to download the training data and place it in the specified folder.
2. **Train the Model**: Run the training script to create and save the model checkpoints.
3. **API Endpoint**: Start the API server to handle prediction requests from RRAT.

## Contributing
Please follow the GitHub flow for contributions:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or inquiries, please contact us at Sumi Means or James Neff.
