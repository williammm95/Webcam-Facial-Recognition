## Webcam-Facial-Recognition-using-PCA

# Objective: Develop a webcam system that recognises specific people using Principle Component Analysis


Fundamental facial features such as eyes, nose and mouth are detected using Haar Classifiers

Data are trained using n=130 pictures, with 5 individual pictures per person.


1. With these n=130 pictures, a reconstructed face is constructed after applying PCA and SVD as shown below.

![Average Face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/intermediate/average_face.jpg)



2. Upon running the program, the webcam will look for faces using opencv2 video capture method.



3. When a face is found, opencv2 converts the face into a column vector and subtracts itself from the average



4. The eclidean distance between feature weights of the test image will be compared with all other weights of all reconstructed images.



5. The least value will be deemed as the matching face.

# Sample pictures will be shown below

![Detection of William's face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/william.png)
![Detection of Joey and an unknown face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/joey.png)
![Detection of CJ's face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/cj.png)
![Detection of Jan's face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/jan.jpg)
![Detection of ST's face](https://github.com/williammm95/Webcam-Facial-Recognition-using-PCA/blob/master/st.png)


# Advantages of PCA
1. Speed - dimensionality reduction
2. Efficiency - does not affect performance accuracy.


# Disadvantages of PCA
1. Highly affected by lighting and intensity condition.


# Ideas for improvement
1. Adding a certain threshold to detect any image in the face database that does not exist
2. Histogram equalisation can be used to boost quality of the recognition process.
3. Alternative classifiers.



