import cv2
import numpy as np
from keras.models import load_model
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.models.builder import builder

# Load pre-trained emotion detection model
emotion_model = load_model('emotion_detection_cnn.h5')

# Load AlphaPose model
cfg = {'cfg':'fast_pose'}
model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

# Set device to CPU
device = 'cpu'

# Load video capture device (webcam)
cap = cv2.VideoCapture(0)

# Define the color codes for each pose joint
joint_colors = [[0, 0, 255], [0, 85, 255], [0, 170, 255], [0, 255, 255], [0, 255, 170], [0, 255, 85], [0, 255, 0], [85, 255, 0], [170, 255, 0], [255, 255, 0], [255, 170, 0], [255, 85, 0], [255, 0, 0]]

# Define the indices of the joints to use for body gesture detection
gesture_joints = [5, 6, 7, 8, 11, 12, 13, 14]

# Define a function to preprocess the frame for emotion detection
def preprocess_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to match the input size of the emotion detection model
    resized = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
    # Reshape the resized frame to match the input shape of the emotion detection model
    reshaped = resized.reshape(1, 48, 48, 1)
    # Normalize the reshaped frame
    normalized = reshaped / 255.0
    return normalized

# Define a function to detect emotions in a frame
def detect_emotions(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Predict the emotions in the preprocessed frame
    predictions = emotion_model.predict(preprocessed_frame)
    # Return the emotion with the highest probability
    return np.argmax(predictions)

# Define a function to detect body gestures in a frame
def detect_gestures(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect poses in the grayscale frame
    poses, scores = model.detect(gray, device)
    # Apply pose non-maximum suppression to remove duplicate poses
    poses, scores = pose_nms(poses, scores)
    # If no poses are detected, return None
    if len(poses) == 0:
        return None
    # Convert the pose heatmaps to coordinates
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    coords = heatmap_to_coord(poses, scores)
    # Extract the gesture joints from the pose coordinates
    gesture_coords = coords[0][gesture_joints]
    # Draw the gesture joints and skeleton on the frame
    for i in range(len(gesture_joints)):
        x, y = int(gesture_coords[i][0]), int(gesture_coords[i][1])
        if x >= 0 and x < frame.shape[1] and y >= 0 and y < frame.shape[0]:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        if i < len(gesture_joints) - 1:
            j = i + 1
            x1, y1 = int(gesture_coords[i][0]), int(gesture_coords[i][1])
            x2, y2 = int(gesture_coords[j][0]), int(gesture_coords[j][1])
            if x1 >= 0 and x1 < frame.shape[1] and y1 >= 0 and y1 < frame.shape[0] and x2 >= 0 and x2 < frame.shape[1] and y2 >= 0 and y2 < frame.shape[0]:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with gesture and emotion detection
    cv2.imshow('Emotion and Gesture Detection', frame)

    # Exit if 'q' is pressed

while True:
     # Read a new frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    preprocess_frame(frame)
    detect_emotions(frame)
    detect_gestures(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
