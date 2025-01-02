import os
from PIL import Image

def crop_and_resize_images(input_path, output_path, img_size=256):
   
    os.makedirs(output_path, exist_ok=True)

    for subdir in os.listdir(input_path):
        subdir_path = os.path.join(input_path, subdir)
        output_subdir_path = os.path.join(output_path, subdir)

        if not os.path.isdir(subdir_path):
            continue

        os.makedirs(output_subdir_path, exist_ok=True)

        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpeg', '.jpg', '.png'))]

        print(f"Processing directory: {subdir_path} with {len(image_files)} images.")

        for i, image_file in enumerate(image_files):
            try:
                img_path = os.path.join(subdir_path, image_file)
                with Image.open(img_path) as img:
                    # Convert to RGB if not already
                    img = img.convert("RGB")

                    # Resize the image
                    img_resized = img.resize((img_size, img_size))

                    # Save the resized image
                    output_file = os.path.join(output_subdir_path, image_file)
                    img_resized.save(output_file)
                    print(f"Resized and saved: {output_file} ({i+1}/{len(image_files)})")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    input_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train"
    output_path = r"C:\Users\sumin\Documents\VScode\ResNet\CNN-model\data\diabetic-retinopathy-detection\train_resized"

    crop_and_resize_images(input_path, output_path)
