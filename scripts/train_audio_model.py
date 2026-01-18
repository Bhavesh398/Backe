import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.audio.audio_dataset import AudioDeepfakeDataset
from backend.audio.audio_model import AudioDeepfakeModel
import os

# ---------------- CONFIG ----------------
DATASET_PATH = "datasets/audio"
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-4
MODEL_PATH = "models/audio_deepfake.pth"
# --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = AudioDeepfakeDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Class counts (for weighting)
real_count = len(os.listdir(f"{DATASET_PATH}/real"))
fake_count = len(os.listdir(f"{DATASET_PATH}/fake"))

print(f"Real: {real_count}, Fake: {fake_count}")

# Weighted loss (VERY IMPORTANT)
weights = torch.tensor(
    [1.0 / real_count, 1.0 / fake_count],
    device=device
)

criterion = nn.CrossEntropyLoss(weight=weights)

# Model
model = AudioDeepfakeModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training
os.makedirs("models", exist_ok=True)
model_saved = False

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    loss_sum = 0.0

    for wavs, labels in loader:
        wavs, labels = wavs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(wavs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss={loss_sum:.2f} | Acc={acc:.2f}%")
    
    # Save model if accuracy >= 95%
    if acc >= 95.0 and not model_saved:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Audio model saved to {MODEL_PATH} (Accuracy: {acc:.2f}%)")
        model_saved = True
        break

# Save model if not already saved
if not model_saved:
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Audio model saved to {MODEL_PATH} (Training completed)")
