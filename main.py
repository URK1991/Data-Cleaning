import cv2
import easygui
import os

# Importing custom modules
from image_cropping import crop_image, crop_with_coords
from text_inpainting import inpaint_text, initialize_pipeline
from file_management import create_clean_dir, get_image_files, save_image

def main():
    # Prompt to select the video folder
    msg = "Select the LUS video folder to clean data"
    title = "LUS DataCleaner"
    button = ['Select Video']
    easygui.buttonbox(msg, title, button)
    
    # Get the path of the selected folder
    folder_dir = easygui.diropenbox()

    # Create a directory to save cleaned images
    sub_dir = folder_dir[31:]  # Extract a subfolder name based on original folder
    save_dir = create_clean_dir('Vids_Cleaned', sub_dir)
    
    # Initialize Keras OCR pipeline
    pipeline = initialize_pipeline()

    # Process each image file in the selected folder
    image_files = get_image_files(folder_dir)
    coords = None

    for i, file in enumerate(image_files):
        impth = os.path.join(folder_dir, file)
        
        if i == 0:
            # First image: manually crop and save the coordinates
            cropped_img, coords = crop_image(impth)
        else:
            # Use the saved coordinates to crop subsequent images
            cropped_img = crop_with_coords(impth, coords)
        
        # Inpaint text in the cropped image
        img_text_removed = inpaint_text(cropped_img, pipeline)
        
        # Save the cleaned image
        save_image(save_dir, file, img_text_removed)
    
if __name__ == "__main__":
    main()
