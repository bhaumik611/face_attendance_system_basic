import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

# ===================== CONFIG =====================
DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 0.45          # Lower = stricter matching
CAMERA_INDEX = 0
FRAME_RESIZE = 0.25       # Speed vs accuracy tradeoff
# ==================================================

print("[INFO] Loading known faces...")

known_encodings = []
known_names = []

# ========== LOAD & ENCODE DATASET ==========
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue   # Skip .DS_Store and junk files

        img_path = os.path.join(person_path, img_name)
        image = face_recognition.load_image_file(img_path)

        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"[INFO] Loaded {len(known_encodings)} face encodings.")

# ========== ATTENDANCE FUNCTION ==========
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # If file does not exist OR is empty → create with headers
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)

    # Prevent duplicate entry for same person on same day
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        df.loc[len(df)] = [name, date, time]
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[ATTENDANCE] {name} marked present.")


# ========== CAMERA START ==========
video = cv2.VideoCapture(CAMERA_INDEX)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Camera started. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)

        if min_distance < THRESHOLD:
            index = np.argmin(distances)
            name = known_names[index]
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Scale back face location
        top, right, bottom, left = face_location
        top *= int(1 / FRAME_RESIZE)
        right *= int(1 / FRAME_RESIZE)
        bottom *= int(1 / FRAME_RESIZE)
        left *= int(1 / FRAME_RESIZE)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2)

    cv2.imshow("Live Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("[INFO] System stopped.")
