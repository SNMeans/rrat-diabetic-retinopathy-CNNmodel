import pandas as pd
import os

def add_rotations_to_undersampled_classes(csv_path, output_csv_path):
    """
    Adds rotated image variations for undersampled classes (1, 2, 3, 4).
    
    Args:
    - csv_path (str): Path to the existing labels CSV.
    - output_csv_path (str): Path to save the updated CSV with rotations.
    """
    # Check if the input CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure the 'image' and 'level' columns exist
    if 'image' not in df.columns or 'level' not in df.columns:
        raise ValueError("The input CSV does not contain the required 'image' or 'level' columns.")
    
    # Classes to augment
    classes_to_augment = [1, 2, 3, 4]
    rotations = ["_rot_90", "_rot_180", "_rot_270"]
    additional_rows = []

    # Filter and augment only the specified classes
    for _, row in df.iterrows():
        image_name, label = row['image'], row['level']
        if label in classes_to_augment:
            for rotation in rotations:
                rotated_image = f"{os.path.splitext(image_name)[0]}{rotation}.jpeg"
                additional_rows.append({"image": rotated_image, "level": label})
    
    # Append the new rows for rotated images
    df = pd.concat([df, pd.DataFrame(additional_rows)], ignore_index=True)
    
    # Save the updated DataFrame to a new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV with rotations for undersampled classes saved to: {output_csv_path}")

if __name__ == "__main__":
    # Use trainLabels_updated.csv instead of trainLabels.csv
    csv_path = "../data/diabetic-retinopathy-detection/trainLabels_updated.csv"
    output_csv_path = "../data/diabetic-retinopathy-detection/trainLabels_balanced.csv"
    
    add_rotations_to_undersampled_classes(csv_path, output_csv_path)

