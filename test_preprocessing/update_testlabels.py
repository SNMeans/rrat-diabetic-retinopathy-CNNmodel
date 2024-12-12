import pandas as pd

# Paths
input_csv = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\testLabels15.csv"
output_csv = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\testLabels15_cleaned.csv"

# Load CSV and remove 'Usage' column
df = pd.read_csv(input_csv)
df = df[['image', 'level']]  # Keep only necessary columns

# Save cleaned CSV
df.to_csv(output_csv, index=False)
print(f"Cleaned CSV saved to: {output_csv}")

