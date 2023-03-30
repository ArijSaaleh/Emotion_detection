import cv2
import numpy as np
from keras.models import load_model
from hrnet import HRNET

# Load pre-trained emotion detection model
emotion_model = load_model('emotion_detection_cnn.h5')

# Load HRNET model for body gesture detection
hrnet = HRNET()

# Define body gesture keypoints
gesture_joints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start capturing video from default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Run HRNET on the frame to detect body gestures
    gesture_coords = hrnet.predict(frame)

    # Run emotion detection on the face using the pre-trained CNN model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=3)
        emotions = emotion_model.predict(face_img)
        max_index = np.argmax(emotions[0])
        emotion_label = emotion_labels[max_index]

        # Draw emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Draw the gesture joints and skeleton on the frame
    for i in range(len(gesture_joints)):
        x, y = int(gesture_coords[gesture_joints[i]][0]), int(gesture_coords[gesture_joints[i]][1])
        if x >= 0 and y >= 0:
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            if i > 0:
                x_, y_ = int(gesture_coords[gesture_joints[i-1]][0]), int(gesture_coords[gesture_joints[i-1]][1])
                cv2.line(frame, (x, y), (x_, y_), (0, 255, 255), 3)

    # Display the resulting frame
    cv2.imshow('Emotion and Gesture Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
