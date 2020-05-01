import os
import numpy as np
import cv2
from face_detection import *


# Original Images in Face Database
f_list = [f for f in os.listdir(face_database_file) if os.path.isfile(os.path.join(face_database_file, f))]

print('Saving grayscale and cropped face images...')

# Detect face, read as grayscale, and resize to fixed size W x H

for file_name in f_list:
    # Read face image as grayscale
    img_gray = cv2.imread(os.path.join(face_database_file, file_name), 0)

    # Detect face
    detected_face_gray, detected_face_coords = detect_face(img_gray)

    # If detected face (grayscale and resized), save to face_database_gray folder
    if detected_face_gray is not None:
        # cv2.imshow('',detected_face_gray)
        # cv2.waitKey(0)
        path_to_new_img = os.path.join(face_database_gray_file, file_name)
        cv2.imwrite(path_to_new_img, detected_face_gray)
    else:
        print("Face on the image %s was not found!" % file_name)

print('Save complete.')