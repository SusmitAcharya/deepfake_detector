"""
inference.py
Run Deepfake Detection on New Media (Images or Videos)
Author: Susmit Acharya | 2025
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import mediapipe as mp
import argparse
import yaml
import time

# ---------------------------------------------------------
# 1. Load Configuration and Model
# ---------------------------------------------------------
CONFIG_PATH = "../config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = "../results/models/mobilenetv3_deepfake.h5"
IMG_SIZE = tuple(cfg["model"]["input_shape"][:2])

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ---------------------------------------------------------
# 2. Helper Function to Process Frame
# ---------------------------------------------------------
def analyze_frame(frame):
    """Detect face, crop, preprocess and predict authenticity."""
    h, w, _ = frame.shape
    results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return 0.5, frame  # No face detected, neutral score

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, IMG_SIZE)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0][0]
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)
        label = f"Authentic: {1-pred:.2f}" if pred < 0.5 else f"Fake: {pred:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return float(pred), frame
    return 0.5, frame

# ---------------------------------------------------------
# 3. Analyze Image or Video
# ---------------------------------------------------------
def run_inference(input_path, output_path=None):
    if input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        frame = cv2.imread(input_path)
        score, output = analyze_frame(frame)
        print(f"[RESULT] Authenticity Score: {1 - score:.3f} (Fake Probability: {score:.3f})")
        if output_path:
            cv2.imwrite(output_path, output)
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            score, annotated = analyze_frame(frame)
            scores.append(score)
            if out is None and output_path:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
            if output_path:
                out.write(annotated)
            cv2.imshow("Deepfake Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        avg_score = np.mean(scores)
        print(f"\n[SUMMARY]")
        print(f"Average Authenticity: {1 - avg_score:.3f}")
        print(f"Average Fake Probability: {avg_score:.3f}")
        print("[INFO] Video analysis complete.")
    else:
        print("[ERROR] Unsupported file format.")

# ---------------------------------------------------------
# 4. Command-Line Interface
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--input", required=True, help="Path to input image/video")
    parser.add_argument("--output", default=None, help="Optional path to save annotated output")
    args = parser.parse_args()

    t0 = time.time()
    run_inference(args.input, args.output)
    print(f"[INFO] Total runtime: {time.time() - t0:.2f}s")
