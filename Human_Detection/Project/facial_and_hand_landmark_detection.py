import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import cv2
import dlib
import mediapipe as mp
import numpy as np

# Initialize Dlib face detector and facial landmark predictor with the specified path
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(r"C:\Users\844ri\OneDrive\Desktop\sec project\shape_predictor_68_face_landmarks.dat")

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale for Dlib and RGB for Mediapipe
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using Dlib
    faces = dlib_detector(gray)
    for face in faces:
        # Get facial landmarks with Dlib
        landmarks = dlib_predictor(gray, face)

        # Bounding box and labels for the eyes
        left_eye_points = [landmarks.part(i) for i in range(36, 42)]
        right_eye_points = [landmarks.part(i) for i in range(42, 48)]
        
        # Draw bounding boxes around eyes
        left_eye_box = cv2.boundingRect(np.array([(p.x, p.y) for p in left_eye_points]))
        right_eye_box = cv2.boundingRect(np.array([(p.x, p.y) for p in right_eye_points]))

        cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), 
                      (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), 
                      (255, 0, 0), 2)
        cv2.putText(frame, "Left Eye", (left_eye_box[0], left_eye_box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), 
                      (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), 
                      (255, 0, 0), 2)
        cv2.putText(frame, "Right Eye", (right_eye_box[0], right_eye_box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Bounding box and label for the nose
        nose_points = [landmarks.part(i) for i in range(27, 36)]
        nose_box = cv2.boundingRect(np.array([(p.x, p.y) for p in nose_points]))
        cv2.rectangle(frame, (nose_box[0], nose_box[1]), 
                      (nose_box[0] + nose_box[2], nose_box[1] + nose_box[3]), 
                      (0, 255, 0), 2)
        cv2.putText(frame, "Nose", (nose_box[0], nose_box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Bounding box and label for the mouth
        mouth_points = [landmarks.part(i) for i in range(48, 68)]
        mouth_box = cv2.boundingRect(np.array([(p.x, p.y) for p in mouth_points]))
        cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), 
                      (mouth_box[0] + mouth_box[2], mouth_box[1] + mouth_box[3]), 
                      (0, 0, 255), 2)
        cv2.putText(frame, "Mouth", (mouth_box[0], mouth_box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Detect hand landmarks with Mediapipe
    hand_results = hands.process(rgb)

    # Draw bounding boxes around hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            ih, iw, _ = frame.shape
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Draw the rectangle and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(frame, "Hand", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the result
    cv2.imshow("Face and Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
