import cv2 

recognizer = cv2.face.EigenFaceRecognizer_create()
detector = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml");
recognizer.read('trainingData.yml')
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
count = 0
print('If you are Chuan Jie, input 0 \nIf you are Janson, input 1 \n'
      'If you are Joey, input 2 \nIf you are William, input 3 \nIf you are Shi Ting, input 4')
names = input('Who are you? Please input your respective number and press Enter: ')

print('Press s to capture your face, or press q to quit the program.')
while True:
    try:
        check, frame = webcam.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        print(check) #prints true as long as the webcam is running
#        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
   
        if key == ord('s'):
            if names == '0':
                name_1 = 'chuan_jie.0.'
                
            elif names == '1':
                name_1 = 'janson.1.'
            
            elif names == '2':
                name_1 = 'joey.2.'
            
            elif names == '3':
                name_1 = 'william.3.'  
            
            elif names == '4':
                name_1 = 'shi_ting.4.'

            name = name_1 + str(count) +'.jpg'
            cv2.imwrite(filename=name, img=frame)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")

            count +=1 
            
        elif key == ord('q'):
            print("Turning camera off...")
            webcam.release()
            print("Camera turned off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
    