import pandas as pd
import opendatasets as od 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotly.offline import iplot, init_notebook_mode
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIL import Image

# Load the trainLabels CSV to link images with labels
labels_df = pd.read_csv('C:/Users/sumin/Documents/VScode/ResNet/rrat-diabetic-retinopathy-CNNmodel/rratmodel/data/diabetic-retinopathy-detection/trainLabels.csv')


# Display the structure of the labels to confirm format
print(labels_df.head())

# Set up lists for class labels and image counts
class_counts = labels_df['level'].value_counts()
Class_name = class_counts.index.tolist()
No_images_per_class = class_counts.values.tolist()

# Plot the number of images per class
sns.barplot(x=Class_name, y=No_images_per_class)
plt.title('Number of Images per Class')
plt.xlabel('Class Label (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)')
plt.ylabel('Number of Images')
plt.show()

# Define the directory containing the images
train_dir = os.path.join('.', 'data', 'diabetic-retinopathy-detection', 'train')

# Visualize 5 images per class based on label data
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
count = 0

for level in Class_name:
    # Get filenames for images in this class
    images_in_class = labels_df[labels_df['level'] == level]['image'].values

    # Limit to 5 images per class for visualization
    for i, image_name in enumerate(images_in_class[:5]):
        img_path = os.path.join(train_dir, f"{image_name}.jpeg")
        
        try:
            img = PIL.Image.open(img_path)
            axs[count][i].imshow(img)
            axs[count][i].set_title(f"Class: {level}")
            axs[count][i].axis('off')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
    count += 1

fig.tight_layout()
plt.show()

# Print total image count and class distribution
total_images = len(labels_df)
class_distribution = labels_df['level'].value_counts()
print(f"Total number of images: {total_images}")
print("Distribution by class:")
print(class_distribution)


# Sample a few images to check dimensions
'''
sample_images = labels_df['image'].values[:10]  # Checking the first 10 images from labels

for img_name in sample_images:
    img_path = os.path.join(train_dir, f"{img_name}.jpeg")  
    try:
        with Image.open(img_path) as img:
            print(f"{img_name} size: {img.size}")  # Output the size of each image
    except FileNotFoundError:
        print(f"File not found: {img_path}") 
'''
# Plot a pie chart showing the percentage of samples per class
plt.figure(figsize=(8, 8))
plt.pie(No_images_per_class, labels=Class_name, autopct='%1.1f%%', startangle=140)
plt.title('Percentage of Samples per Class')
plt.show()

# Data Augmentation and Data Generators
# Convert numeric labels to strings for ImageDataGenerator
labels_df['Labels'] = labels_df['level'].astype(str)  

# Create a DataFrame that links images and labels
labels_df['image_path'] = labels_df['image'].apply(lambda x: os.path.join(train_dir, f"{x}.jpeg"))

# Check column names and adjust if necessary
print(labels_df.columns)  # Check the columns before renaming

# Optionally rename if only 3 columns are expected, adjust accordingly
# For example, if 'image' and 'level' are the relevant columns, plus the new 'image_path'
labels_df = labels_df[['image', 'Labels', 'image_path']]
labels_df.columns = ['Image', 'Labels', 'image_path']

# Verify the DataFrame structure
print(labels_df.head())

# Define training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15  # Reserve 15% of the training data for validation
)

# Define test data generator (only rescaling, no augmentation for test data)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up generators for training, validation, and testing data
train_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col="image_path",
    y_col="Labels",
    target_size=(256, 256), #This will resize all images to 256X256 px
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32, #Images will be processed in batches of 32
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col="image_path",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    subset='validation'
)

# Define test generator with no augmentation
test_generator = test_datagen.flow_from_dataframe(
    labels_df,
    x_col="image_path",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32
)

# Display one batch of images
batch = next(train_generator)
images, labels = batch
plt.figure(figsize=(12, 12))
for i in range(9):  # Show 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title("Class: " + str(np.argmax(labels[i])))
    plt.axis('off')
plt.show()