import pandas as pd

# Path to trainLabels.csv
csv_path = 'data/diabetic-retinopathy-detection/trainLabels.csv'

# Load the CSV
labels_df = pd.read_csv(csv_path)

# Add .jpeg extension to the image column
labels_df['image'] = labels_df['image'] + '.jpeg'

# Save the updated CSV
updated_csv_path = 'data/diabetic-retinopathy-detection/trainLabels_updated.csv'
labels_df.to_csv(updated_csv_path, index=False)

print(f"Updated CSV saved to {updated_csv_path}")
