import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from backend.models.image_model import DeepfakeImageModel
from backend.datasets.image_dataset import ImageDeepfakeDataset

# ---------------- CONFIG ----------------
EPOCHS = 12
BATCH_SIZE = 8
LR = 3e-5
DATASET_PATH = "datasets/images"
MODEL_SAVE_PATH = "models/image_deepfake.pth"
TRAIN_SPLIT = 0.8  # 80/20 train/validation split
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset & 80/20 Split
dataset = ImageDeepfakeDataset(DATASET_PATH)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Dataset split -> Train: {train_size}, Validation: {val_size}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model (from scratch - no pre-loaded weights)
model = DeepfakeImageModel()
model.to(device)

# Loss & Optimizer
import os

# ---- CLASS WEIGHTED LOSS (CRITICAL FIX) ----
num_real = len(os.listdir(os.path.join(DATASET_PATH, "real")))
num_fake = len(os.listdir(os.path.join(DATASET_PATH, "fake")))

print(f"Class count -> Real: {num_real}, Fake: {num_fake}")

weights = torch.tensor(
    [1.0 / num_real, 1.0 / num_fake],
    device=device
)

criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer (classifier only is even better, see note below)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training Loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = (train_correct / train_total) * 100
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = (val_correct / val_total) * 100
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

# Save final model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Final model saved to {MODEL_SAVE_PATH}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
