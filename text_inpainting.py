import cv2
import math
import numpy as np
import keras_ocr

# Utility function to calculate the midpoint between two points
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)

# Function to detect text using Keras OCR and inpaint it from the image
def inpaint_text(img, pipeline):
    # Recognize text (and corresponding regions)
    prediction_groups = pipeline.recognize([img])
    
    # Define a mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    # Iterate over the detected text boxes
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        
        # Calculate line thickness based on distance between corners
        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        
        # Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return inpainted_img

# Initialize the OCR pipeline
def initialize_pipeline():
    return keras_ocr.pipeline.Pipeline()
