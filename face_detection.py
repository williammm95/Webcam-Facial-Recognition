import numpy as np
import cv2

# Width and Height of face image
W, H = 100, 100

# Face Database folder, Face Database Grayscale folder, "Intermediate" folder to save intermediate results
face_database_file, face_database_gray_file, intermediate_file = 'face_database', 'face_database_gray', 'intermediate'

face_Cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')


# Functions for face detection

def detect_face(img_gray):
    global face_Cascade

    detected_face_gray_resized, detected_face_coords = None, None
    detected_faces = face_Cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

    # Find the coordinates of the largest face
    largest_coords = largest_face(detected_faces)

    if largest_coords is not None:
        (x, y, w, h) = largest_coords
        detected_face_coords = largest_coords  # copy location of largeest face (x,y,w,h)

        cropped_face = img_gray[y:y + h, x:x + w]  # crop only the face
        detected_face_gray_resized = cv2.resize(cropped_face, (W, H))  # resize the largest face to W x H

    return detected_face_gray_resized, detected_face_coords


def calculate_area(rect):
    ''' Calculate the area of detected face '''
    x, y, w, h = rect
    area = w * h
    return area


def largest_face(detected_faces):
    ''' Find the largest face among all detected faces '''
    # No faces found
    if len(detected_faces) == 0:
        print('No faces detected!')
        return None

    areas = [calculate_area(face) for face in detected_faces]
    max_index = np.argmax(areas)
    largest_detected_face = detected_faces[max_index]
    return largest_detected_face
