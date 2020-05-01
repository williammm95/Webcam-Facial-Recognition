from PIL import Image
import os
import numpy as np
import cv2
import tkinter as tk
from face_detection import *

# Default values for projections, eigenfaces and mean face
from ipython_genutils.py3compat import xrange

training_proj, eigenfaces, average_face_flatten = None, None, None

# Face Database Grayscale folder with grayscale faces (used for SVD and PCA)
f_list = [f_name for f_name in os.listdir(face_database_gray_file) if
          os.path.isfile(os.path.join(face_database_gray_file, f_name))]

# Total number of face images
n = sum([True for f in f_list])


## Functions ##

root = None
new_face_content = None
name_var = None


def onSaveNewImage(*args):
    face_name = name_var.get() + '.jpg'
    face_path = os.path.join(face_database_file, face_name)
    cv2.imwrite(face_path, new_face_content)

    if root is not None:
        root.destroy()


def draw_rectangles(img, rectangles):
    for (x, y, w, h) in rectangles:
        pt1, pt2 = (x, y), (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, color=(255,0,0))


def euclidean_dist(vector1, vector2):
    ''' Euclidean distance between vectors '''
    dist = np.sqrt(np.sum((vector1 - vector2) ** 2))
    return dist


def norm(array):
    # Use norm1 to normalize
    return array / np.linalg.norm(array)


def compute_svd_pca():
    global W, H
    face_width, face_height = (W, H)

    # Create a vector for all faces
    face_vector = np.array([cv2.imread(os.path.join(face_database_gray_file, filename), 0).flatten() for filename in f_list])

    # Compute average face
    fave = np.mean(face_vector, 0)

    # Subtract the average face from each image before performing SVD and PCA
    X = face_vector - fave

    print("Finding SVD of data matrix")
    # Decompose the mean-centered matrix into three parts

    U, S, Vt = np.linalg.svd(X.transpose(), full_matrices=False)
    V = Vt.T

    # Sort principal components by descending order of the singular values
    ind = np.argsort(S)[::-1]
    U, S, V = U[:, ind], S[ind], V[:, ind]
    eigenfaces = U

    # Print Dimensions
    print("face_vector:", face_vector.shape)
    print("U:", U.shape)
    print("Sigma:", S.shape)
    print("V^T:", Vt.shape)

    # Weights is an n x n matrix
    weights = np.dot(X, eigenfaces)  # TODO: Maybe swap + .T to eigenfaces

    # Some intermediate save:
    save_average_face = True
    if save_average_face:
        # Save average face
        average_face = fave.reshape(face_width, face_height)
        cv2.imwrite(os.path.join(intermediate_file, 'average_face.jpg'), average_face)

    save_eigenvectors = False
    if save_eigenvectors:
        print("Saving eigenvectors...")
        for i in xrange(n):
            f_name = os.path.join(intermediate_file, 'eigenvector_%s.png' % i)
            im = U[:, i].reshape(face_width, face_height)
            cv2.imwrite(f_name, im)

    save_reconstructed = True
    if save_reconstructed:
        k = 30
        print('\n', 'Save the reconstructed images based on only "%s" eigenfaces' % k)
        for img_id in range(n):
            # for k ranging from 1 to total + 1:
            reconstructed_face = fave + np.dot(weights[img_id, :k], eigenfaces[:, :k].T)
            reconstructed_face.shape = (face_width, face_height)  # transform vector to initial image size
            cv2.imwrite(os.path.join(intermediate_file, 'img_reconstr_%s_k=%s.png' % (f_list[img_id], k)), reconstructed_face)


    # Projected training images into PCA subspace as yn=weights or Yn = E.T * (Xn - average_face)
    training_proj = weights
    average_face_flatten = fave

    return training_proj, eigenfaces, average_face_flatten


def recognize_face(face_gray):
    global training_proj, eigenfaces, average_face_flatten, f_list

    # Convert face to vector
    face_gray_flatten = face_gray.flatten()

    # Compute SVD + PCA if not already computed
    if None in [training_proj, eigenfaces, average_face_flatten]:
        training_proj, eigenfaces, average_face_flatten = compute_svd_pca()

    # Subtract average face from the target face
    print(average_face_flatten.shape)
    test_f = face_gray_flatten - average_face_flatten

    # Project test image into PCA space
    test_proj = np.dot(eigenfaces.T, test_f)

    # Calculate distance between one test image and all other training images
    d = np.zeros((n, 1))
    for i in range(n):
        d[i] = euclidean_dist(training_proj[i], test_proj)
    min_dist_id = d.argmin()


    # Open matched image and show filename
    found_face_filename = f_list[min_dist_id]
    found_face_img = cv2.imread(os.path.join(face_database_gray_file, found_face_filename))

    print('File name is "%s"' % found_face_filename)
    cv2.imshow(found_face_filename, found_face_img)
    cv2.waitKey(0)


## Start of main program ##

if __name__ == '__main__':
    # global root, new_face_content, name_var

    video_cam_recognition = True
    single_img_recognition = False

    if video_cam_recognition:
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            exit('Web camera is not connected')

        print('Available commands:')
        print('\n', 'Press "Enter" to capture current frame as image from web camera and add it to database')
        print('\n', 'Press "Space" to recognize face from current frame from web camera')
        print('\n', 'Press "Q" to close and skip to labelled face \n')

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detecting face
                detected_face_gray, detected_face_coords = detect_face(frame_gray)

                frame_with_mask = None
                if detected_face_gray is not None:
                    # Mark found face
                    mask = np.zeros_like(frame)  # init mask
                    draw_rectangles(mask, [detected_face_coords])
                    frame_with_mask = cv2.add(frame, mask)

                # Show current frame
                frame_to_show = frame_with_mask if frame_with_mask is not None else frame
                cv2.imshow('Video', frame_to_show)  # show either frame (if face isn't detected or frame with mask)

                # Process picture when SPACE is pressed
                k = cv2.waitKey(1)
                if k % 256 == 32 and detected_face_gray is not None:
                    # Start recognition process
                    print('Starting the recognition process...')
                    recognize_face(detected_face_gray)
                    print('Recognition complete.')

                elif k & 0xFF in [ord('\r'), ord('\n')]:
                    print('Enter key pressed - save image')

                    if detected_face_gray is not None:
                        # Run save new image form
                        root = tk.Tk()
                        new_face_content = frame
                        name_var = tk.StringVar(root)

                        tk.Label(root, text='Enter your name').grid(row=0)
                        tk.Entry(root, textvariable=name_var).grid(row=1)
                        tk.Button(root, text='Save image', command=onSaveNewImage).grid(row=2)

                        root.mainloop()

                    else:
                        print('Face is not detected.')

                # Exit
                elif k & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print('Please wait...')

        # Release capture
        video_capture.release()
        cv2.destroyAllWindows()


# Open webcam again to recognize and label encoded or unknown faces
        
import cv2
import face_recognition

img_counter = 0

video_capture = cv2.VideoCapture(0)

# Load sample pictures of known faces and label them
Chuanjie_image = face_recognition.load_image_file("C:/Users/Joey/Desktop/EE4208_Assignment1/face_database_gray/chuan_jie.0.0.jpg")
Chuanjie_encoding = face_recognition.face_encodings(Chuanjie_image)[0]

Janson_image = face_recognition.load_image_file("C:/Users/Joey/Desktop/EE4208_Assignment1/face_database_gray/janson.1.0.jpg")
Janson_encoding = face_recognition.face_encodings(Janson_image)[0]

Joey_image = face_recognition.load_image_file("C:/Users/Joey/Desktop/EE4208_Assignment1/face_database_gray/joey.2.3.jpg")
Joey_encoding = face_recognition.face_encodings(Joey_image)[0]

William_image = face_recognition.load_image_file("C:/Users/Joey/Desktop/EE4208_Assignment1/face_database_gray/william.3.0.jpg")
William_encoding = face_recognition.face_encodings(William_image)[0]

Shiting_image = face_recognition.load_image_file("C:/Users/Joey/Desktop/EE4208_Assignment1/face_database_gray/shi_ting.4.0.jpg")
Shiting_encoding = face_recognition.face_encodings(Shiting_image)[0]

# Create array of known face encodings
known_face_encodings = [
    Chuanjie_encoding,
    Janson_encoding,
    Joey_encoding,
    William_encoding,
    Shiting_encoding
]

# Create array of known face names
known_face_names = [
    "Chuan Jie",
    "Janson",
    "Joey",
    "William",
    "Shi Ting"
]

path = 'C:/Users/Joey/Desktop/EE4208_Assignment1/'
faceDetect = cv2.CascadeClassifier(path + 'haarcascade_frontalface.xml')
video_capture = cv2.VideoCapture(0)


print('\n', 'Press "Esc" to close the program.')

while True:
    # Get a single frame of video
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = faceDetect.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # OpenCV uses BGR, face_recognition uses RGB
    # Convert the image from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # Check for match of known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Match found in known_face_encodings
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label face
        cv2.rectangle(frame, (left, bottom - 1), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 1, bottom), font, 0.8, (255, 255, 255), 1)

    # Display face with name
    cv2.imshow('Video', frame)

    if not ret:
        break

    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Closing the program...")
        break
    

video_capture.release()
cv2.destroyAllWindows()

print("Program closed.")
