import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
TRAIN_DIR = "../data/diabetic-retinopathy-detection/train_augmented"
VALIDATION_DIR = "../data/diabetic-retinopathy-detection/validation"
MODEL_DIR = "../models/"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Data generators
def create_generators(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Important for correct evaluation
    )

    return train_generator, val_generator

# Build ResNet50 model
def build_resnet50_model(num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Train the model
def train_and_evaluate_model():
    train_generator, val_generator = create_generators(TRAIN_DIR, VALIDATION_DIR)
    num_classes = train_generator.num_classes

    model = build_resnet50_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // 32,
        validation_steps=val_generator.samples // 32
    )

    # Save the model with a unique timestamp
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_OUTPUT = os.path.join(MODEL_DIR, f"resnet50_diabetic_retinopathy_{TIMESTAMP}.h5")
    model.save(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")

    # Evaluate the model on the validation set
    val_generator.reset()  # Ensure correct order for predictions
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # Classification Report
    class_names = list(val_generator.class_indices.keys())
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names)

    return history

if __name__ == "__main__":
    train_and_evaluate_model()

