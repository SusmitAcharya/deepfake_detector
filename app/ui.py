import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import tensorflow as tf
import uvicorn

# Import your architecture builder
from model.mobilenet_transfer import build_mobilenet_transfer

# ---------------------------------------------------------
# 1. Paths & Config
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "results", "models", "mobilenet_v3_final.h5")

app = FastAPI(title="Deepfake Detector")

# static & templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

IMG_SIZE = (224, 224)
LEARNING_RATE = 3e-4  # same as used in training

# ---------------------------------------------------------
# 2. Load model safely (rebuild architecture + load weights)
# ---------------------------------------------------------
print(f"[INFO] Loading model weights from: {MODEL_PATH}")
try:
    model = build_mobilenet_transfer(input_shape=IMG_SIZE + (3,), learning_rate=LEARNING_RATE)
    model.load_weights(MODEL_PATH)
    print("[INFO] Model loaded successfully (weights only).")
except Exception as e:
    print(f"[ERROR] Could not load weights: {e}")
    model = None

# ---------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------
def preprocess_image(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------------
# 4. Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "result": "Model not loaded"})

    data = await file.read()
    arr = preprocess_image(data)
    pred = model.predict(arr)[0][0]

    label = "Deepfake" if pred > 0.5 else "Authentic"
    confidence = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)
    color = "#e63946" if label == "Deepfake" else "#06d6a0"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": f"{label} ({confidence}%)", "color": color}
    )

# ---------------------------------------------------------
# 5. Run the app
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.ui:app", host="0.0.0.0", port=8000, reload=True)
