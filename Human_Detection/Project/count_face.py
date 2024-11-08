import cv2
import face_recognition

# Initialize the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the video feed
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convert the image from BGR color (OpenCV format) to RGB color (face_recognition format)
    rgb_frame = frame[:, :, ::-1]
    
    # Detect all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Count the number of faces detected
    face_count = len(face_locations)
    
    # Display the face count on the video frame
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw rectangles around each face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Display the frame with face count and rectangles
    cv2.imshow('Face Counter', frame)
    
    # Break out of the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
