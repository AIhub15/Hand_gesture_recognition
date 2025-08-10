import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import zipfile
import requests
from tqdm import tqdm

app = Flask(__name__)

# Configuration
DATASET_URL = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/master.zip"
DATASET_PATH = "dataset"
MODEL_PATH = "gesture_model.h5"
IMG_SIZE = 64

# Gesture mapping
GESTURES = {
    0: {"name": "Closed Fist", "description": "Hand closed in a fist"},
    1: {"name": "Pointing Up", "description": "Index finger pointing up"},
    2: {"name": "Victory Sign", "description": "Index and middle finger up (peace sign)"},
    3: {"name": "Three Fingers", "description": "Three fingers extended"},
    4: {"name": "Four Fingers", "description": "Four fingers extended"},
    5: {"name": "Open Hand", "description": "All five fingers extended"},
    6: {"name": "Thumbs Up", "description": "Thumb extended upward"},
    7: {"name": "OK Sign", "description": "Thumb and index finger forming a circle"},
    8: {"name": "Gun Sign", "description": "Index finger and thumb extended like a gun"},
    9: {"name": "Shaka Sign", "description": "Thumb and pinky extended"}
}

CLASSES = list(GESTURES.keys())

# Preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    reshaped = np.reshape(norm, (1, IMG_SIZE, IMG_SIZE, 1))
    return reshaped

# Download and unzip dataset (if required)
def download_dataset():
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        print("Downloading dataset...")
        r = requests.get(DATASET_URL, stream=True)
        zip_path = os.path.join(DATASET_PATH, "dataset.zip")
        with open(zip_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_PATH)
        os.remove(zip_path)

# Load dataset
def load_data():
    download_dataset()
    path = os.path.join(DATASET_PATH, "Sign-Language-Digits-Dataset-master/Dataset")
    data = []
    labels = []
    for label in CLASSES:
        dir_path = os.path.join(path, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = to_categorical(labels)
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save model
def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
    model.fit(X_train, y_train, batch_size=32, epochs=15,
              validation_data=(X_test, y_test), callbacks=callbacks)
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = train_model()

# Open webcam
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')  # Make sure templates/index.html exists

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        size = min(h, w)
        offset_w = (w - size) // 2
        offset_h = (h - size) // 2
        cropped = frame[offset_h:offset_h+size, offset_w:offset_w+size]

        processed = preprocess_frame(cropped)
        pred = model.predict(processed, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        gesture = GESTURES[CLASSES[class_id]]

        return jsonify({
            'gesture_name': gesture['name'],
            'gesture_description': gesture['description'],
            'confidence': float(confidence),
            'class_id': int(CLASSES[class_id])
        })
    return jsonify({'error': 'No frame captured'})

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        size = min(h, w)
        offset_w = (w - size) // 2
        offset_h = (h - size) // 2
        cropped = frame[offset_h:offset_h+size, offset_w:offset_w+size]

        processed = preprocess_frame(cropped)
        pred = model.predict(processed, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        gesture = GESTURES[CLASSES[class_id]]

        cv2.putText(frame, f"Gesture: {gesture['name']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, gesture['description'], (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.rectangle(frame, (offset_w, offset_h), (offset_w+size, offset_h+size), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
