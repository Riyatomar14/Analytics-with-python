import cv2
import numpy as np
import os

# Ensure the images directory exists
images_dir = "images"
trainer_file = "trainer.yml"

def train_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_paths = []
    ids = []
    faces = []

    # Gather images and corresponding IDs from the folder
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # Check if image was loaded correctly
                if gray_img is not None:
                    faces.append(gray_img)
                    id = int(os.path.basename(root).split("_")[1])  # Extract user ID from folder name
                    ids.append(id)
                else:
                    print(f"Warning: Could not read image {path}")
    
    # Check if we have images to train on
    if len(faces) == 0:
        print("No images found for training.")
        return

    # Train and save the recognizer
    recognizer.train(faces, np.array(ids))
    recognizer.save(trainer_file)
    print("Model trained and saved as trainer.yml.")

# Run the training function
train_faces()

