import torch
import os
from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import load_image
from backend.video.frame_extractor import extract_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepfakeImageModel()
model.load_state_dict(torch.load("models/image_deepfake.pth", map_location=device))
model.to(device)
model.eval()

def predict_video(video_path):
    frame_dir = "temp_frames"
    num_frames = extract_frames(video_path, frame_dir)

    fake_probs = []

    with torch.no_grad():
        for file in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, file)
            img = load_image(frame_path).unsqueeze(0).to(device)
            out = model(img)
            prob = torch.softmax(out, dim=1)[0][1].item()
            fake_probs.append(prob)

    avg_fake = sum(fake_probs) / len(fake_probs)

    return {
        "frames_analyzed": len(fake_probs),
        "avg_fake_probability": round(avg_fake * 100, 2),
        "verdict": "FAKE" if avg_fake >= 0.6 else "REAL"
    }
