import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Paths
MODEL_PATH = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\models\resnet50_diabetic_retinopathy.h5"
VALIDATION_DIR = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\validation"

# Load the trained model
model = load_model(MODEL_PATH)

# Prepare the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),  # Match training size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate on validation data
val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Get true labels and predictions
true_labels = val_generator.classes
predictions = model.predict(val_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=list(val_generator.class_indices.keys())))

# Generate confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

