import os
from PIL import Image

def rotate_and_resize_images(folder_path, output_folder_path=None):
    # Set the desired resolution
    target_width, target_height = 1280, 720

    # Create output folder if it doesn't exist
    if output_folder_path and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # Check if the image is vertical
                    if height > width:
                        # Rotate the image 90 degrees to make it horizontal
                        img = img.rotate(90, expand=True)
                        print(f"Rotated {filename} by 90 degrees to make it horizontal.")
                    
                    # Resize the image to 720p resolution
                    resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
                    
                    # Determine the save path
                    if output_folder_path:
                        save_path = os.path.join(output_folder_path, filename)
                    else:
                        save_path = file_path
                    
                    # Save the processed image
                    resized_img.save(save_path)
                    print(f"Resized and saved {filename} to {save_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Specify the folder path containing images
input_folder_path = 'D:/Downloads/DIV2K_train_HR'

# Optionally, specify an output folder path
output_folder_path = 'D:/Projects/DeepDeblur-PyTorch/converted'  # Set to None to overwrite original images

# Resize images
rotate_and_resize_images(input_folder_path, output_folder_path)
