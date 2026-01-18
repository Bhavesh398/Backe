import torch
from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = DeepfakeImageModel()
checkpoint = torch.load('models/image_deepfake.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
print(f"  - Real accuracy: {checkpoint['val_acc_real']:.2f}%")
print(f"  - Fake accuracy: {checkpoint['val_acc_fake']:.2f}%")
print()

# Test image
img = load_image('test.jpg').unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)[0]

print(f"Raw output: {output}")
print(f"Probabilities:")
print(f"  Real: {probs[0].item():.4f} ({probs[0].item()*100:.2f}%)")
print(f"  Fake: {probs[1].item():.4f} ({probs[1].item()*100:.2f}%)")
print(f"\nPrediction: {'REAL' if probs[0] > probs[1] else 'FAKE'}")
print(f"Confidence: {abs(probs[0].item() - probs[1].item())*100:.2f}%")
