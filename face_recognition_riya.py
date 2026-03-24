import cv2
import os
import numpy as np
import pickle
import threading

FACES_DIR      = "known_faces"
ENCODINGS_FILE = "face_data.pkl"
os.makedirs(FACES_DIR, exist_ok=True)

known_faces = {}

def save_faces():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_faces, f)

def load_faces():
    global known_faces
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            known_faces = pickle.load(f)

def register_face(name):
    load_faces()
    cap      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    samples  = []
    count    = 0

    print(f"Registering face for {name}. Look at camera...")

    while count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            samples.append(face_roi.flatten())
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {count}/10",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("Register Face - Press Q to quit", frame)
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        known_faces[name] = np.mean(samples, axis=0)
        save_faces()
        print(f"Face registered for {name}!")
        return True
    print("No face detected!")
    return False

def recognize_face():
    load_faces()
    if not known_faces:
        return None, 0

    cap      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attempts = 0

    while attempts < 50:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi  = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            face_flat = face_roi.flatten().astype(np.float64)

            best_name = None
            best_conf = 0

            for name, stored in known_faces.items():
                stored = stored.astype(np.float64)
                norm1  = np.linalg.norm(face_flat)
                norm2  = np.linalg.norm(stored)
                if norm1 == 0 or norm2 == 0:
                    continue
                similarity = np.dot(face_flat, stored) / (norm1 * norm2)
                if similarity > best_conf:
                    best_conf = similarity
                    best_name = name

            if best_conf > 0.92:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,
                    f"{best_name} ({best_conf:.0%})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(1000)
                cap.release()
                cv2.destroyAllWindows()
                return best_name, best_conf

        cv2.putText(frame, "Looking for face...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv2.imshow("Face Recognition", frame)
        cv2.waitKey(100)
        attempts += 1

    cap.release()
    cv2.destroyAllWindows()
    return None, 0

if __name__ == "__main__":
    print("Riya Face Recognition (OpenCV)")
    print("=" * 40)
    print("1. Register your face")
    print("2. Recognize face")
    choice = input("Choose (1/2): ")

    if choice == "1":
        name = input("Enter your name: ")
        register_face(name)
    elif choice == "2":
        name, conf = recognize_face()
        if name:
            print(f"Hello {name}! Confidence: {conf:.0%}")
        else:
            print("Face not recognized!")