import face_recognition
import cv2
import pickle
from server import app, run_server, status
import threading

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

with open("known_faces.dat", "rb") as f:
    known_faces, known_names = pickle.load(f)

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Cannot open camera")
    exit()

def update_status(new_status):
    global status
    status = new_status
    print(f"{status}")

def face_recognition_loop():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        locations = face_recognition.face_locations(frame, model=MODEL)
        encodings = face_recognition.face_encodings(frame, locations)

        if len(encodings) == 0:
            update_status("No face detected")
        else:
            intruder_alert = True
            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    intruder_alert = False
                    update_status(f"Match Found: {match}")

                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = [0, 255, 0]
                    cv2.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)

                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)
                    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(frame, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

            if intruder_alert:
                update_status("INTRUDER ALERT")
                print("Intruder Alert")

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=run_server).start()
    face_recognition_loop()