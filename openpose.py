import cv2
import numpy as np
from keras.models import load_model
import pyopenpose as op

# Load pre-trained model
model = load_model('emotion_detection_cnn.h5')

# Initialize OpenPose
params = dict()
params["model_folder"] = "../openpose/models"
params["face"] = True
params["hand"] = False
params["body"] = 1
params["disable_multi_thread"] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Initialize camera capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Loop over frames from the camera
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    # Detect body keypoints
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    keypoints = datum.poseKeypoints

    # If no body keypoints detected, skip the frame
    if keypoints.size == 0:
        continue

    # Extract face from the frame
    face_keypoints = keypoints[0, :5, :2]
    face_rect = cv2.boundingRect(np.array([face_keypoints]))
    face_img = frame[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]]

    # If face image is too small, skip the frame
    if face_img.shape[0] < 64 or face_img.shape[1] < 64:
        continue

    # Preprocess the face image
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)

    # Predict emotion using the pre-trained CNN model
    emotion_probs = model.predict(face_img)[0]
    emotion_label = emotion_labels[emotion_probs.argmax()]

    # Draw emotion label on the frame
    cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw body keypoints and skeleton on the frame
    op_frame = datum.cvOutputData
    op_frame = cv2.resize(op_frame, (frame.shape[1], frame.shape[0]))
    op_frame = cv2.cvtColor(op_frame, cv2.COLOR_BGR2GRAY)
    op_frame = cv2.merge((op_frame, op_frame, op_frame))
    op_frame = cv2.addWeighted(op_frame, 0.5, frame, 0.5, 0)
    for i in range(keypoints.shape[1]):
        for j in range(keypoints.shape[2]):
            x, y = int(keypoints[0, i, j]), int(keypoints[0, i, j+1])
            if x >= 0 and x < frame.shape[1] and y >= 0 and y < frame.shape[0]:
                cv2.circle(op_frame, (x, y), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if keypoints[0, partA, 2] and keypoints[0, partB, 2]:
                xA, yA = int(keypoints[0, partA, 0]), int(keypoints[0, partA, 1])
                xB, yB = int(keypoints[0, partB, 0]), int(keypoints[0, partB, 1])
                cv2.line(op_frame, (xA, yA), (xB, yB), (0, 255, 255), 3)

    # Display the resulting frame
    cv2.imshow('frame', op_frame)

    # Emotion Detection
    if face_img is not None:
        # Pre-process the image
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = np.reshape(face_img, [1, face_img.shape[0], face_img.shape[1], 1])

        # Predict the emotion
        emotion_preds = model.predict(face_img)[0]
        emotion_label = EMOTIONS[np.argmax(emotion_preds)]
        print('Emotion:', emotion_label)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Clean up resources
cap.release()
cv2.destroyAllWindows()