import sys
import asyncio
import time
import json
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Windows-only fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_DIR = BASE_DIR / "models"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

model = joblib.load(MODEL_DIR / "best_model.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
COORDS = 3
FEATURES_PER_FRAME = (POSE_LANDMARKS + 2 * HAND_LANDMARKS) * COORDS
WINDOW_SIZE = 30
INFERENCE_INTERVAL = 4.0

feature_buffer = deque(maxlen=WINDOW_SIZE)

def extract_landmarks(results):
    features = []

    def extract(block, count):
        if block:
            for lm in block.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * count)

    extract(results.pose_landmarks, POSE_LANDMARKS * COORDS)
    extract(results.left_hand_landmarks, HAND_LANDMARKS * COORDS)
    extract(results.right_hand_landmarks, HAND_LANDMARKS * COORDS)

    return np.array(features, dtype=np.float32)

def landmarks_to_dict(results):
    def pack(block):
        return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in block.landmark] if block else []
    return {
        "pose": pack(results.pose_landmarks),
        "left_hand": pack(results.left_hand_landmarks),
        "right_hand": pack(results.right_hand_landmarks)
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    collecting = False
    next_inference_time = None
    last_results = None

    try:
        while True:
            message = await ws.receive()

            if "text" in message:
                if message["text"] == "START":
                    collecting = True
                    feature_buffer.clear()
                    next_inference_time = time.time() + INFERENCE_INTERVAL
                    continue

                if message["text"] == "STOP":
                    collecting = False
                    feature_buffer.clear()
                    continue

            if not collecting:
                continue

            if "bytes" in message:
                frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                last_results = results
                feature_buffer.append(extract_landmarks(results))

            if next_inference_time and time.time() >= next_inference_time:
                X = np.array(feature_buffer)

                if len(X) >= WINDOW_SIZE:
                    idx = np.linspace(0, len(X) - 1, WINDOW_SIZE).astype(int)
                    X = X[idx]
                else:
                    X = np.vstack([X, np.zeros((WINDOW_SIZE - len(X), FEATURES_PER_FRAME))])

                X = X.flatten().reshape(1, -1)
                pred = model.predict(X)[0]
                label = encoder.inverse_transform([pred])[0]

                await ws.send_text(json.dumps({
                    "label": label,
                    "landmarks": landmarks_to_dict(last_results)
                }))

                feature_buffer.clear()
                next_inference_time = time.time() + INFERENCE_INTERVAL + 1.0

    except WebSocketDisconnect:
        pass
