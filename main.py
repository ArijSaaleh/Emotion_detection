import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained model
model = load_model('emotion_detection_cnn.h5')

# Define emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Set up video capture
cap = cv2.VideoCapture(0)
while True:
    # Read a frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize pixel values to be between 0 and 1
        roi_gray = np.array(roi_gray, dtype='float') / 255.0

        # Reshape for input to CNN model
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Make prediction with CNN model
        preds = model.predict(roi_gray)[0]
        label = EMOTIONS[preds.argmax()]

        # Draw rectangle around face and label emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
