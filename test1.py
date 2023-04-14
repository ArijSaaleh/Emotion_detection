import cv2

# initialize face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# initialize video capture device
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the camera
    ret, frame = cap.read()
    
    # convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # iterate over the faces and draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # display the resulting frame
    cv2.imshow('frame', frame)
    
    # check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
