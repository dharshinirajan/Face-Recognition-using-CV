# Import necessary libraries
import cv2
import numpy as np
import pickle
import os

# Set up video capture from default camera
url = 0

# File names for storing face data and names
names_file = "names.pickle"
face_data_file = "face_data.pickle"

# Initialize video capture and face detection
video = cv2.VideoCapture(url)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_data = []
i = 0

# Input the name of the person
name = input('Enter the name: ')

# Main loop for capturing face data
while True:
    # Read frame from the video capture
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Crop and resize face region
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Store resized face images
        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)
        i += 1
        
        # Display face count on frame and draw rectangle around face
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    # Display frame with annotations
    cv2.imshow("Frame", frame)
    
    # Wait for 'q' key or until 100 face samples are collected
    k = cv2.waitKey(1)
    if k == ord('q') or len(face_data) == 100:
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()

# Convert face data to numpy array and reshape
face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

# Store names associated with collected face data
if not os.path.exists(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Store face data
if not os.path.exists(face_data_file):
    with open(face_data_file, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(face_data_file, 'rb') as f:
        face_data_existing = pickle.load(f)
    face_data_combined = np.append(face_data_existing, face_data, axis=0)
    with open(face_data_file, 'wb') as f:
        pickle.dump(face_data_combined, f)
