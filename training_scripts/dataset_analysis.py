import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def analyze_dataset(csv_path):
    """
    Analyze the dataset for balance and other metrics.
    
    Args:
    - csv_path (str): Path to the CSV file containing dataset labels.
    """
    df = pd.read_csv(csv_path)
    
    # Check the distribution of labels
    label_counts = df['level'].value_counts()

    # Class labels
    class_labels = ['0 No DR', '1 Mild DR', '2 Moderate DR', '3 Severe DR', '4 Proliferative DR']

    
   # Colors for each bar
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#DDA0DD']  # Light pastel colors
    
    # Plot a bar chart for label distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, label_counts, color=colors, edgecolor='black')

    # Add title and axis labels
    plt.title("Level of DR (Diabetic Retinopathy) vs Frequency", fontsize=16, fontweight='bold')
    plt.xlabel("Disease Level", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)

    # Customize tick parameters
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate the bars with their heights
    for i, count in enumerate(label_counts):
        plt.text(i, count + 0.02 * max(label_counts), str(count), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Add Pie Chart for Label Distribution
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Class Distribution of Diabetic Retinopathy", fontsize=16, fontweight='bold')
    plt.show()
    
    # Print basic stats
    print("Label Distribution:")
    print(label_counts)

if __name__ == "__main__":
    csv_path = "../data/diabetic-retinopathy-detection/trainLabels_balanced.csv"
    analyze_dataset(csv_path)

