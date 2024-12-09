import os
import shutil
from PIL import Image, ImageOps


def copy_class_0(input_path, output_path):
    """
    Copies the folder for class '0' from input to output without applying any augmentation.
    """
    class_0_input_path = os.path.join(input_path, '0')
    class_0_output_path = os.path.join(output_path, '0')

    # Create the output directory for class 0 if it doesn't exist
    os.makedirs(class_0_output_path, exist_ok=True)

    # Copy all images from input class 0 to output class 0
    for img_file in os.listdir(class_0_input_path):
        if img_file.endswith('.jpeg'):
            src = os.path.join(class_0_input_path, img_file)
            dest = os.path.join(class_0_output_path, img_file)
            shutil.copy(src, dest)
            print(f"Copied {src} to {dest}")


def rotate_images(input_path, output_path, degrees_of_rotation, classes_to_augment):
    """
    Rotates images in the input directory by the specified degrees
    and saves them to the output directory. Only applies to specified classes.
    """
    for class_folder in os.listdir(input_path):
        if class_folder not in classes_to_augment:
            print(f"Skipping class {class_folder} for rotation.")
            continue

        class_input_path = os.path.join(input_path, class_folder)
        class_output_path = os.path.join(output_path, class_folder)
        os.makedirs(class_output_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_input_path) if f.endswith('.jpeg')]
        for img_file in image_files:
            img_path = os.path.join(class_input_path, img_file)
            with Image.open(img_path) as img:
                rotated_img = img.rotate(degrees_of_rotation)
                new_file = f"{os.path.splitext(img_file)[0]}_rot{degrees_of_rotation}.jpeg"
                rotated_img.save(os.path.join(class_output_path, new_file))
                print(f"Saved rotated image: {os.path.join(class_output_path, new_file)}")


def mirror_images(input_path, output_path, classes_to_augment):
    """
    Mirrors (flips horizontally) images in the input directory
    and saves them to the output directory. Only applies to specified classes.
    """
    for class_folder in os.listdir(input_path):
        if class_folder not in classes_to_augment:
            print(f"Skipping class {class_folder} for mirroring.")
            continue

        class_input_path = os.path.join(input_path, class_folder)
        class_output_path = os.path.join(output_path, class_folder)
        os.makedirs(class_output_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_input_path) if f.endswith('.jpeg')]
        for img_file in image_files:
            img_path = os.path.join(class_input_path, img_file)
            with Image.open(img_path) as img:
                mirrored_img = ImageOps.mirror(img)
                new_file = f"{os.path.splitext(img_file)[0]}_mir.jpeg"
                mirrored_img.save(os.path.join(class_output_path, new_file))
                print(f"Saved mirrored image: {os.path.join(class_output_path, new_file)}")


if __name__ == "__main__":
    input_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_resized"
    output_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_augmented"

    # Classes to augment (exclude class 0)
    classes_to_augment = ['1', '2', '3', '4']

    # Step 1: Copy class 0 without augmentation
    copy_class_0(input_path, output_path)

    # Step 2: Perform rotation
    rotate_images(input_path, output_path, degrees_of_rotation=90, classes_to_augment=classes_to_augment)
    rotate_images(input_path, output_path, degrees_of_rotation=180, classes_to_augment=classes_to_augment)
    rotate_images(input_path, output_path, degrees_of_rotation=270, classes_to_augment=classes_to_augment)

    # Step 3: Perform mirroring
    mirror_images(input_path, output_path, classes_to_augment=classes_to_augment)

