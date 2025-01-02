import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(csv_path):
    """
    Analyze the dataset for balance and other metrics.
    
    Args:
    - csv_path (str): Path to the CSV file containing dataset labels.
    """
    df = pd.read_csv(csv_path)
    
    # Check the distribution of labels
    label_counts = df['level'].value_counts()
    
    # Plot a bar chart for label distribution
    plt.figure(figsize=(8, 6))
    label_counts.plot(kind='bar')
    plt.title("Dataset Label Distribution")
    plt.xlabel("Disease Level")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.show()
    
    # Print basic stats
    print("Label Distribution:")
    print(label_counts)

if __name__ == "__main__":
    csv_path = "../data/diabetic-retinopathy-detection/trainLabels_balanced.csv"
    analyze_dataset(csv_path)

