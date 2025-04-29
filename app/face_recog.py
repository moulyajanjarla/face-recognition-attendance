import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(known_faces_path="known_faces"):
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_faces_path):
        image = face_recognition.load_image_file(f"{known_faces_path}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def recognize_face(frame, known_encodings, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for face_encoding, face_location in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return known_names[best_match_index]
    return None
