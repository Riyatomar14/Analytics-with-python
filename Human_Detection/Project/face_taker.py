import cv2
import json
import os

# Ensure that 'images' directory exists
os.makedirs("images", exist_ok=True)

# Ensure names.json exists
if not os.path.exists("names.json"):
    with open("names.json", "w") as f:
        json.dump({}, f)

def take_pictures(user_name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    count = 0
    user_id = len(os.listdir('images')) + 1

    # Create user directory if it doesn't exist
    user_folder = f"images/user_{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    
    # Add new entry to names.json
    with open("names.json", "r+") as f:
        names = json.load(f)
        names[str(user_id)] = user_name
        f.seek(0)
        json.dump(names, f)
    
    print("Capturing images. Please make sure your face is centered.")
    while count < 30:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.imwrite(f"{user_folder}/image_{count}.jpg", gray[y:y+h, x:x+w])
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if count >= 30:
                break
        
        cv2.imshow("Taking Pictures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"Captured 30 images for {user_name}.")

# Run the function to capture images
user_name = input("Enter user name: ")
take_pictures(user_name)
