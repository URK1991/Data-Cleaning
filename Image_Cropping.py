import cv2

# Function for selecting a region of interest (ROI) in the image and cropping it
def crop_image(pth):
    img = cv2.imread(pth)
    imagedraw = cv2.selectROI(img)  # User-defined ROI
    cropped_image = img[int(imagedraw[1]):int(imagedraw[1]+imagedraw[3]), 
                        int(imagedraw[0]):int(imagedraw[0]+imagedraw[2])]
    
    # Displaying the cropped image as output
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cropped_image, imagedraw

# Function to crop an image with pre-defined coordinates (based on the first image)
def crop_with_coords(pth, imagedraw):
    img = cv2.imread(pth)
    cropped_image = img[int(imagedraw[1]):int(imagedraw[1]+imagedraw[3]), 
                        int(imagedraw[0]):int(imagedraw[0]+imagedraw[2])]
    return cropped_image
