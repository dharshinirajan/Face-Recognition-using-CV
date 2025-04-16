import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from imutils import face_utils
import dlib

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function for attendance monitoring
def attendance_monitoring():
    # Get the path of the project folder
    project_folder = os.path.dirname(os.path.abspath(__file__))
    # Load names and face data
    names_file = os.path.join(project_folder, 'names.pickle')
    face_data_file = os.path.join(project_folder, 'face_data.pickle')
    with open(names_file, 'rb') as f:
        labels = pickle.load(f)
    with open(face_data_file, 'rb') as f:
        face_data = pickle.load(f)
    
    # Initialize classifiers and constants
    mugam_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(face_data, labels)
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0

    emotions = ["Sad", "Happy", "Cry", "Neutral"]

    # Define the folder for storing attendance files
    attendance_folder = os.path.join(project_folder, 'attendance_csv')

    # Initialize video capture
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        faces = mugam_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in mugam:
            # Process each detected face
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            emotion = emotions[3]
            if len(eyes) >= 2:
                eye_region = face_roi[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
                avg_intensity = cv2.mean(eye_region)[0]
                
                if avg_intensity < 60:  
                    emotion = emotions[0] if avg_intensity < 40 else emotions[2]
                elif avg_intensity > 100:
                    emotion = emotions[1]
            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = shape_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            TOTAL=0
            # Calculate drowsiness
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            # Get current timestamp and drowsiness status
            ts = time.time()
            drowsiness = "Yes" if COUNTER >= EYE_AR_CONSEC_FRAMES else "No"

            # Display information on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 0, 0), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, "Drowsiness: {}".format(drowsiness), (x, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

        # Show frame
        cv2.imshow("Attendance Monitoring", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

attendance_monitoring()
