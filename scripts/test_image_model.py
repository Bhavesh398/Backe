import torch
from backend.preprocessing.image import load_image
from backend.models.image_model import DeepfakeImageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = DeepfakeImageModel()
model.to(device)
model.eval()

# Load image
img = load_image("test.jpg", device=device)

# Inference
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

print("Probabilities:", probs.cpu().numpy())
print("Fake probability:", probs[0][1].item())
