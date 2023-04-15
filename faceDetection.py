import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model

# Load pre-trained model for emotion detection
model = load_model('emotion_detection_cnn.h5')

# Initialize MTCNN for face detection
detector = MTCNN()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)

    # Loop through each face and detect emotions
    for face in faces:
        # Extract face region of interest
        x, y, w, h = face['box']
        roi = frame[y:y+h, x:x+w]

        # Preprocess face image for emotion detection
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Predict emotion using pre-trained model
        preds = model.predict(roi)[0]
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        emotion_label = emotion_labels[np.argmax(preds)]

        # Draw emotion label on face bounding box
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy window
cap.release()
cv2.destroyAllWindows()
