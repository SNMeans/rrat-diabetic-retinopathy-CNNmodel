# utils.py
# Purpose: Contains reusable functions for loading, preprocessing, data generators, and dataset visualization.
# Used by main.py for dataset exploration and by train_resnet50.py for model training.
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load and preprocess data
def load_and_preprocess_data(train_labels_path, train_dir):
    labels_df = pd.read_csv(train_labels_path)
    labels_df['Labels'] = labels_df['level'].astype(str)
    labels_df['image_path'] = labels_df['image'].apply(lambda x: os.path.join(train_dir, f"{x}.jpeg"))
    return labels_df


# Generate data augmentation pipelines
def create_generators(labels_df):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15  # 15% validation split
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=labels_df,
        x_col="image_path",
        y_col="Labels",
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=labels_df,
        x_col="image_path",
        y_col="Labels",
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=labels_df,
        x_col="image_path",
        y_col="Labels",
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


# Plot the number of images per class
def plot_class_distribution(labels_df):
    class_counts = labels_df['level'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Number of Images per Class')
    plt.xlabel('Class Label (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)')
    plt.ylabel('Number of Images')
    plt.show()


# Plot sample images from each class
def visualize_images(labels_df, train_dir):
    class_counts = labels_df['level'].value_counts().index.tolist()

    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    count = 0

    for level in class_counts:
        images_in_class = labels_df[labels_df['level'] == level]['image'].values
        for i, image_name in enumerate(images_in_class[:5]):
            img_path = os.path.join(train_dir, f"{image_name}.jpeg")
            try:
                img = Image.open(img_path)
                axs[count][i].imshow(img)
                axs[count][i].set_title(f"Class: {level}")
                axs[count][i].axis('off')
            except FileNotFoundError:
                print(f"File not found: {img_path}")
        count += 1

    fig.tight_layout()
    plt.show()
