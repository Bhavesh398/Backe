from fastapi import FastAPI, UploadFile, File
import torch
import shutil
import os
import logging

from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import load_image
from backend.video.frame_extractor import extract_frames
from backend.audio.audio_inference import predict_audio
from fastapi.middleware.cors import CORSMiddleware

# ---------------- APP ----------------
app = FastAPI(title="Deepfake Forensics API")

# Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
        "https://hack-w-ins-project-r18a.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake-api")

# ---------------- CONFIG ----------------
IMAGE_MODEL_PATH = "models/image_deepfake.pth"
UPLOAD_DIR = "uploads"
FRAME_DIR = "temp_frames"

TEMPERATURE = 1.4
FAKE_THRESHOLD = 0.70

IMAGE_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mp4", ".avi", ".mov")
AUDIO_EXT = (".wav", ".mp3")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ---------------- LOAD IMAGE MODEL ----------------
image_model = DeepfakeImageModel().to(device)

if os.path.exists(IMAGE_MODEL_PATH):
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        image_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        image_model.load_state_dict(checkpoint)
    image_model.eval()
else:
    logger.warning(f"Image model not found at {IMAGE_MODEL_PATH}. Image inference may not work.")

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"message": "Deepfake Forensics API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }

# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ================= IMAGE =================
    if filename.endswith(IMAGE_EXT):
        image = load_image(file_path).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = image_model(image)
            probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]

        fake_prob = probs[1].item()
        real_prob = probs[0].item()

        if fake_prob >= FAKE_THRESHOLD:
            verdict = "FAKE"
        elif fake_prob <= (1 - FAKE_THRESHOLD):
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        return {
            "type": "image",
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "verdict": verdict,
            "confidence": round(abs(fake_prob - 0.5) * 200, 2)
        }

    # ================= VIDEO =================
    if filename.endswith(VIDEO_EXT):
        for f in os.listdir(FRAME_DIR):
            os.remove(os.path.join(FRAME_DIR, f))

        frames = extract_frames(file_path, FRAME_DIR, every_n_frames=10)
        if frames == 0:
            return {"error": "No frames extracted"}

        fake_probs = []

        with torch.no_grad():
            for frame in os.listdir(FRAME_DIR):
                img = load_image(os.path.join(FRAME_DIR, frame)).unsqueeze(0).to(device)
                logits = image_model(img)
                probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]
                fake_probs.append(probs[1].item())

        fake_probs.sort()
        trim = int(0.1 * len(fake_probs))
        trimmed = fake_probs[trim:-trim] if len(fake_probs) > 20 else fake_probs
        avg_fake = sum(trimmed) / len(trimmed)

        if avg_fake >= FAKE_THRESHOLD:
            verdict = "FAKE"
        elif avg_fake <= (1 - FAKE_THRESHOLD):
            verdict = "REAL"
        else:
            verdict = "FAKE"

        return {
            "type": "video",
            "frames_analyzed": len(fake_probs),
            "fake_probability": round(avg_fake * 100, 2),
            "real_probability": round((1 - avg_fake) * 100, 2),
            "verdict": verdict,
            "confidence": round(abs(avg_fake - 0.5) * 200, 2)
        }

    # ================= AUDIO =================
    if filename.endswith(AUDIO_EXT):
        fake_prob = predict_audio(file_path)

        if fake_prob >= 0.75:
            verdict = "FAKE"
        elif fake_prob <= 0.30:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        return {
            "type": "audio",
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round((1 - fake_prob) * 100, 2),
            "verdict": verdict,
            "confidence": round(abs(fake_prob - 0.5) * 200, 2)
        }

    return {"error": "Unsupported file format"}
