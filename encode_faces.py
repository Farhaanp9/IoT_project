import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"  # Can be "hog" or "cnn"

print("Loading known faces...")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)
       
        if len(encodings) > 0:
            encoding = encodings[0]
            known_faces.append(encoding)
            known_names.append(name)
        else:
            print(f"No faces found in {filename}")

with open("known_faces.dat", "wb") as f:
    pickle.dump((known_faces, known_names), f)

print("Encoding complete!")