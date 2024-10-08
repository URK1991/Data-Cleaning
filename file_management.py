import os
import cv2

# Function to create a directory for cleaned images
def create_clean_dir(base_dir, sub_dir):
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# Function to save an image to a specified path
def save_image(save_dir, file_name, image):
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Function to get the list of files in a directory
def get_image_files(folder_dir):
    return [f for f in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, f))]
