import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
TRAIN_DIR = "../data/diabetic-retinopathy-detection/train"
VALIDATION_DIR = "../data/diabetic-retinopathy-detection/validation"
MODEL_OUTPUT = "../models/resnet50_diabetic_retinopathy.h5"

# Data generators
def create_generators(train_dir, validation_dir):
    """
    Creates data generators for training and validation datasets.
    Applies augmentation to the training data for better generalization.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
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
        class_mode='categorical'
    )

    return train_generator, val_generator

# Build ResNet50 model
def build_resnet50_model(num_classes):
    """
    Builds and returns a ResNet50-based model with custom classification layers.
    Pre-trained weights from ImageNet are used for feature extraction.
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)  # Use Global Average Pooling instead of Flatten
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze base model layers to retain pre-trained features
    for layer in base_model.layers:
        layer.trainable = False

    return model

# Train the model
def train_model():
    """
    Trains the ResNet50 model on the specified training and validation datasets.
    Saves the trained model to the specified output path.
    """
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

    # Save the model
    model.save(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")

    return history

if __name__ == "__main__":
    train_model()

