import os
from PIL import Image, ImageOps

def rotate_images(input_path, output_path, angles):
    """
    Rotates images by the specified angles and saves them to the output path.

    Parameters:
        input_path (str): Path to the input images.
        output_path (str): Path to save the rotated images.
        angles (list): List of angles to rotate.
    """
    for class_folder in os.listdir(input_path):
        class_input_path = os.path.join(input_path, class_folder)
        class_output_path = os.path.join(output_path, class_folder)
        os.makedirs(class_output_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_input_path) if f.endswith('.jpeg')]
        for img_file in image_files:
            img_path = os.path.join(class_input_path, img_file)
            img = Image.open(img_path)
            for angle in angles:
                rotated_img = img.rotate(angle)
                new_file = f"{os.path.splitext(img_file)[0]}_rot{angle}.jpeg"
                rotated_img.save(os.path.join(class_output_path, new_file))
                print(f"Saved rotated image: {os.path.join(class_output_path, new_file)}")

if __name__ == "__main__":
    input_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_resized"
    output_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_augmented"

    angles = [90, 180, 270]
    rotate_images(input_path, output_path, angles)

