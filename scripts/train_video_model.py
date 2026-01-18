import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.video.temporal_dataset import VideoTemporalDataset
from backend.video.temporal_model import CNNLSTMDeepfake

# ---------------- CONFIG ----------------
DATASET_PATH = "datasets/videos"
FRAMES_PER_CLIP = 16
BATCH_SIZE = 2          # keep small for GPU memory
EPOCHS = 5
LR = 1e-4
MODEL_SAVE_PATH = "models/video_temporal.pth"
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = VideoTemporalDataset(
    root_dir=DATASET_PATH,
    frames_per_clip=FRAMES_PER_CLIP
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,# 🔥 Windows-safe
)

# Model
model = CNNLSTMDeepfake()
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    total = 0
    correct = 0
    epoch_loss = 0.0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = (correct / total) * 100
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Acc: {acc:.2f}%"
    )

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Temporal video model saved to {MODEL_SAVE_PATH}")
