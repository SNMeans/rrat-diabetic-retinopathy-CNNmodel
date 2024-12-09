import os
import matplotlib.pyplot as plt
import seaborn as sns

# Directory path for train_augmented
TRAIN_AUGMENTED_DIR = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_augmented"

def count_images_per_class(directory):
    """
    Count the number of images in each class directory.
    """
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):  # Check if it's a directory
            num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = num_images
    return class_counts

def plot_class_distribution(class_counts):
    """
    Plot the distribution of image counts per class.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=counts, palette="Blues")
    plt.title("Box Plot of Image Counts Per Class")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.show()

    # Bar plot for clarity
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts, palette="Blues")
    plt.title("Image Counts Per Class")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.show()

if __name__ == "__main__":
    # Count images
    class_counts = count_images_per_class(TRAIN_AUGMENTED_DIR)
    print("Image counts per class:", class_counts)

    # Plot distribution
    plot_class_distribution(class_counts)

